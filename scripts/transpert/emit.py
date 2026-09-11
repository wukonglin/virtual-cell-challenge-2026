"""Turn a per-perturbation log2 fold-change into a submittable count matrix.

All six vcc2026 metrics are functions of the per-perturbation pseudobulk plus the
Wilcoxon DE call on top of it, so a model's real output is one log2FC vector per
perturbation; making 400 cells out of that is a fixed step every model shares.

The emission follows cell-eval2's own baseline construction
(``cell_eval2.baseline._emit_scaled_resample``): draw control cells with replacement
and scale them gene-wise by ``2**lfc``. Multiplicative scaling keeps counts
non-negative without the clipping an additive shift would need, and conditional on
the control pool it gives ``E[group mean] = ctrl * 2**lfc`` exactly on every gene the
controls reach. Genes that are zero across the whole control pool are unreachable --
no scaling brings them up -- which is a real ceiling on this family of models, not a
bug in the emission.

Counts must be integers (``allow_fractional_counts: false``), so the scaled values are
rounded *stochastically*: ``floor(v) + (u < frac(v))``. Deterministic rounding biases
every fold change under 1.5x toward zero, which is most of the signal.
"""
from __future__ import annotations

import json
import numpy as np
import scipy.sparse as sp

CONTROL = "non-targeting"


def load_context(path: str):
    """Read a challenge context file -> (CSR counts, gene names, obs dict of arrays)."""
    import h5py

    with h5py.File(path, "r") as f:
        shape = tuple(f["X"].attrs["shape"])
        X = sp.csr_matrix(
            (f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]), shape=shape
        )
        vi = f["var"].attrs["_index"]
        node = f[f"var/{vi}"]
        # anndata writes a nullable string array as a {mask, values} group
        genes = (node["values"] if isinstance(node, __import__("h5py").Group) else node)[:]
        genes = genes.astype(str)
        obs = {}
        for k in f["obs"].attrs["column-order"]:
            o = f[f"obs/{k}"]
            if isinstance(o, __import__("h5py").Group):
                obs[k] = o["categories"][:].astype(str)[o["codes"][:]]
            else:
                obs[k] = o[:]
    return X, genes, obs


def pseudobulk(X: sp.csr_matrix) -> np.ndarray:
    """Mean raw counts per gene over cells."""
    return np.asarray(X.mean(axis=0)).ravel()


def emit(
    ctrl_X: sp.csr_matrix,
    lfc: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
    max_scale: float = 64.0,
) -> sp.csr_matrix:
    """``n_cells`` cells for one perturbation: resampled controls scaled by ``2**lfc``.

    ``max_scale`` caps the multiplier. A predicted +10 log2FC on a gene the controls
    barely express would otherwise put thousands of counts on one gene and blow the
    per-cell budget; real CRISPRi responses do not reach there.
    """
    scale = np.clip(np.exp2(np.asarray(lfc, dtype=np.float64)), 0.0, max_scale)
    src = rng.integers(0, ctrl_X.shape[0], size=n_cells)
    out = ctrl_X[src].copy().astype(np.float64)
    out.data *= scale[out.indices]
    # stochastic rounding: unbiased, and keeps sub-1.5x fold changes alive
    floor = np.floor(out.data)
    out.data = floor + (rng.random(out.data.size) < (out.data - floor))
    out.data = out.data.astype(np.float32)
    out.eliminate_zeros()
    return out


def build_submission(
    ctrl_X: sp.csr_matrix,
    perts: list[str],
    lfc_by_pert: dict[str, np.ndarray],
    n_cells: int = 400,
    seed: int = 0,
    max_scale: float = 64.0,
):
    """Stack every perturbation for one context -> (CSR, target_gene labels)."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    n_genes = ctrl_X.shape[1]
    zero = np.zeros(n_genes)
    for p in perts:
        blocks.append(emit(ctrl_X, lfc_by_pert.get(p, zero), n_cells, rng, max_scale))
        labels.extend([p] * n_cells)
    return sp.vstack(blocks, format="csr"), np.array(labels)


def write_h5ad(path: str, X: sp.csr_matrix, target_gene: np.ndarray, genes: np.ndarray):
    """Write the submission h5ad: CSR float32 counts, obs/target_gene, var/_index."""
    import anndata as ad
    import pandas as pd

    a = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(
            {"target_gene": pd.Categorical(target_gene)},
            index=pd.Index([f"cell_{i}" for i in range(X.shape[0])], name=None),
        ),
        var=pd.DataFrame(index=pd.Index(genes, name=None)),
    )
    a.write_h5ad(path, compression="gzip")
    return a


def submission_report(X: sp.csr_matrix, n_cells_expected: int) -> dict:
    """The numbers `vcc prep` will check, so we can fail here instead of at upload."""
    per_cell = np.asarray(X.sum(axis=1)).ravel()
    return {
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "stored_entries": int(X.nnz),
        "stored_entries_cap": 4_750_000_000,
        "entries_per_cell": float(X.nnz / max(X.shape[0], 1)),
        "max_counts_per_cell": float(per_cell.max()) if per_cell.size else 0.0,
        "median_counts_per_cell": float(np.median(per_cell)) if per_cell.size else 0.0,
        "all_integer": bool(np.all(X.data == np.floor(X.data))),
        "any_negative": bool((X.data < 0).any()),
        "rows_expected": n_cells_expected,
    }


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


class SubmissionWriter:
    """Stream a submission to h5ad one perturbation at a time.

    The full submission is 360,000 cells x 18,533 genes at roughly 6,000 stored entries
    per cell -- about 2.2e9 nonzeros, which is over int32's 2.147e9 and would silently
    push a scipy CSR onto int64 indices mid-build, doubling an already 17 GB matrix. It
    also does not need to exist all at once: every perturbation's block is independent.
    So write straight into resizable HDF5 datasets and keep memory at one block.
    """

    def __init__(self, path: str, genes, n_genes: int | None = None):
        import h5py

        self.path = path
        self.genes = np.asarray(genes).astype(str)
        self.n_genes = int(n_genes or len(self.genes))
        self.f = h5py.File(path, "w")
        g = self.f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        self.data = g.create_dataset("data", (0,), maxshape=(None,), dtype="float32",
                                     chunks=(1 << 20,), compression="gzip",
                                     compression_opts=1)
        self.indices = g.create_dataset("indices", (0,), maxshape=(None,), dtype="int32",
                                        chunks=(1 << 20,), compression="gzip",
                                        compression_opts=1)
        self.indptr = g.create_dataset("indptr", (1,), maxshape=(None,), dtype="int64",
                                       chunks=(1 << 16,))
        self.indptr[0] = 0
        self.n_rows = 0
        self.nnz = 0
        self.labels: list[str] = []
        self.contexts: list[str] = []
        self.max_row_total = 0.0

    def add(self, block: sp.csr_matrix, pert: str, context: str):
        block = block.tocsr()
        n, k = block.shape
        if k != self.n_genes:
            raise ValueError(f"block has {k} genes, expected {self.n_genes}")
        self.data.resize((self.nnz + block.nnz,))
        self.indices.resize((self.nnz + block.nnz,))
        self.data[self.nnz:] = block.data.astype(np.float32)
        self.indices[self.nnz:] = block.indices.astype(np.int32)
        self.indptr.resize((self.n_rows + n + 1,))
        self.indptr[self.n_rows + 1:] = block.indptr[1:].astype(np.int64) + self.nnz
        totals = np.asarray(block.sum(axis=1)).ravel()
        if totals.size:
            self.max_row_total = max(self.max_row_total, float(totals.max()))
        self.nnz += block.nnz
        self.n_rows += n
        self.labels.extend([pert] * n)
        self.contexts.extend([context] * n)

    def close(self) -> dict:
        import h5py

        f = self.f
        f["X"].attrs["shape"] = np.array([self.n_rows, self.n_genes], dtype="int64")
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
        _write_frame(f, "obs",
                     {"target_gene": np.array(self.labels),
                      "context": np.array(self.contexts)},
                     np.array([f"cell_{i}" for i in range(self.n_rows)]))
        _write_frame(f, "var", {}, self.genes)
        for name in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            grp = f.create_group(name)
            grp.attrs["encoding-type"] = "dict"
            grp.attrs["encoding-version"] = "0.1.0"
        report = {
            "n_cells": self.n_rows,
            "n_genes": self.n_genes,
            "stored_entries": int(self.nnz),
            "stored_entries_cap": 4_750_000_000,
            "entries_per_cell": round(self.nnz / max(self.n_rows, 1), 1),
            "max_counts_per_cell": self.max_row_total,
            "n_perturbations": len(set(self.labels)),
            "contexts": sorted(set(self.contexts)),
        }
        f.close()
        return report


def _write_frame(f, name: str, cols: dict, index: np.ndarray):
    """Minimal anndata 0.2.0 dataframe encoding: categoricals for every string column."""
    import h5py

    g = f.create_group(name)
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["_index"] = "_index"
    # h5py has no native encoding for an empty object array, which is what var's
    # (always empty) column list is -- name the string dtype explicitly for both.
    g.attrs["column-order"] = np.array(list(cols), dtype=h5py.string_dtype())
    g.create_dataset("_index", data=index.astype(object),
                     dtype=h5py.string_dtype(), compression="gzip")
    for k, v in cols.items():
        cats, codes = np.unique(np.asarray(v).astype(str), return_inverse=True)
        cg = g.create_group(k)
        cg.attrs["encoding-type"] = "categorical"
        cg.attrs["encoding-version"] = "0.2.0"
        cg.attrs["ordered"] = False
        cg.create_dataset("categories", data=cats.astype(object),
                          dtype=h5py.string_dtype())
        cg.create_dataset("codes", data=codes.astype(np.int32), compression="gzip")
