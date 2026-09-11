"""Reduce the PerturBench corpus to per-(cell line, perturbation) CPM pseudobulks.

The corpus files ship as ``log1p(normalize_total(target_sum=9715))``, which is exactly
invertible for our purpose: ``expm1(x)`` is already a per-cell profile summing to 9715,
so ``mean_cells(expm1(x)) * 1e6/9715`` is the arithmetic-mean CPM pseudobulk that
vcc2026's DE recipe is built on. The raw per-cell depth is never needed.

Both baselines that transfer to an unseen cell context read from these tables: log2 fold
changes are the closest thing to a cell-line-invariant currency for a knockdown, which is
the premise Team Outlier's TransPert (VCC-2025, 3rd) built on and the target Team
XLearning's (2nd) network regresses.

Gene spaces differ per source -- Nadig's are HVG-filtered to ~9k, McFaline's carries
40,775 -- so each table keeps its own gene axis and callers align by symbol.
"""
from __future__ import annotations

import os
import numpy as np
import h5py

TARGET_SUM = 9715.0
CONTROL = "control"


def _codes(f: h5py.File, key: str):
    o = f[f"obs/{key}"]
    if isinstance(o, h5py.Group):
        return o["categories"][:].astype(str), o["codes"][:]
    v = o[:].astype(str)
    cats, codes = np.unique(v, return_inverse=True)
    return cats, codes.astype(np.int32)


def _var_index(f: h5py.File) -> np.ndarray:
    node = f[f"var/{f['var'].attrs['_index']}"]
    node = node["values"] if isinstance(node, h5py.Group) else node
    return node[:].astype(str)


def build_table(path: str, out: str, pert_col: str = "condition",
                chunk: int = 40_000, verbose: bool = True,
                keep: dict[str, str] | None = None,
                cell_line: str | None = None) -> dict:
    """Stream one corpus h5ad -> ``out.npz`` of CPM pseudobulks, one row per condition.

    ``keep`` restricts to cells matching every ``{obs column: value}`` pair, which is how
    a multi-cell-line or multi-treatment file (McFaline-Figueroa's is both) is split into
    one comparable table per cell line -- pooling them would average three lineages and
    five drugs into a single fictitious "control".
    """
    with h5py.File(path, "r") as f:
        genes = _var_index(f)
        cats, codes = _codes(f, pert_col)
        line = cell_line or (_codes(f, "cell_type")[0][0] if "cell_type" in f["obs"] else "?")
        dataset = _codes(f, "dataset")[0][0] if "dataset" in f["obs"] else os.path.basename(path)
        n_cells, n_genes = (int(x) for x in f["X"].attrs["shape"])
        mask = np.ones(n_cells, dtype=bool)
        for col, val in (keep or {}).items():
            c, k = _codes(f, col)
            mask &= (c[k] == val)
        indptr = f["X/indptr"][:]
        acc = np.zeros((len(cats), n_genes), dtype=np.float64)
        n_per = np.bincount(codes[mask], minlength=len(cats))
        for s in range(0, n_cells, chunk):
            e = min(s + chunk, n_cells)
            lo, hi = indptr[s], indptr[e]
            # expm1 undoes the log1p; values are then per-cell counts scaled to TARGET_SUM
            dat = np.expm1(f["X/data"][lo:hi].astype(np.float64))
            idx = f["X/indices"][lo:hi]
            ptr = indptr[s:e + 1] - lo
            rows = np.repeat(np.arange(s, e), np.diff(ptr))
            sel = mask[rows]
            np.add.at(acc, (codes[rows[sel]], idx[sel]), dat[sel])
            if verbose:
                print(f"    {e:,}/{n_cells:,}", end="\r", flush=True)
    if verbose:
        print()
    cpm = acc / np.maximum(n_per, 1)[:, None] * (1e6 / TARGET_SUM)
    np.savez_compressed(
        out, cpm=cpm.astype(np.float32), perts=cats, genes=genes,
        n_cells=n_per, cell_line=str(line), dataset=str(dataset),
    )
    return {"path": path, "cell_line": str(line), "dataset": str(dataset), "keep": keep or {},
            "n_conditions": int((n_per > 0).sum()), "n_genes": int(n_genes),
            "n_cells": int(mask.sum()), "has_control": bool(CONTROL in set(cats))}


class RefTable:
    """One cell line's CPM pseudobulks, with log2FC on demand."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path, allow_pickle=False)
        self.cpm = z["cpm"]
        self.perts = z["perts"].astype(str)
        self.genes = z["genes"].astype(str)
        self.n_cells = z["n_cells"]
        self.cell_line = str(z["cell_line"])
        self.dataset = str(z["dataset"])
        self.key = f"{self.dataset}/{self.cell_line}"
        self._row = {p: i for i, p in enumerate(self.perts)}
        if CONTROL not in self._row:
            raise ValueError(f"{npz_path} has no '{CONTROL}' condition")
        self.control_cpm = self.cpm[self._row[CONTROL]].astype(np.float64)
        self._s2 = None

    def __repr__(self):
        return (f"RefTable({self.key}, {len(self.perts)} conditions, "
                f"{len(self.genes)} genes)")

    def targets(self, min_cells: int = 20) -> list[str]:
        """Single-gene knockdowns with enough cells to give a usable pseudobulk."""
        return [p for p in self.perts
                if p != CONTROL and "+" not in p
                and self.n_cells[self._row[p]] >= min_cells]

    def lfc(self, pert: str, pseudocount: float = 1.0,
            shrink: bool = True) -> np.ndarray:
        """log2 fold change vs this cell line's own controls, in CPM space.

        The pseudocount is in CPM, not the scorer's 1e-9: at 1e-9 a gene seen in the
        perturbed pseudobulk but not the control returns a fold change of ~30 log2 units,
        which is sampling noise promoted to the loudest signal in the table.

        ``shrink`` applies a per-gene James-Stein factor ``s2/(s2+v)``. It is not
        optional in spirit: across four reference cell lines the strongest correlate of a
        perturbation's measured response magnitude is its **cell count**, at Spearman
        -0.37 to -0.68 -- an artifact, not biology. A knockdown measured on 30 cells looks
        far more dramatic than the same knockdown measured on 300, and averaging raw
        log2FCs across references would let the thinnest, noisiest arms shout loudest.

        The sampling variance of a log2 pseudobulk mean for a gene at ``mu`` expected
        counts per cell over ``n`` cells is about ``(1/n)(1/mu)/ln(2)^2`` (Poisson,
        delta method); the control arm contributes the same with its own ``n``.
        """
        raw = self._raw_lfc(pert, pseudocount)
        if not shrink:
            return raw
        v = self._noise_var(float(self.n_cells[self._row[pert]]))
        return raw * (self.signal_var / (self.signal_var + v))

    def _noise_var(self, n_pert: float) -> np.ndarray:
        """Per-gene sampling variance of a log2 pseudobulk ratio at ``n_pert`` cells."""
        n_p = max(n_pert, 1.0)
        n_c = max(float(self.n_cells[self._row[CONTROL]]), 1.0)
        mu = np.maximum(self.control_cpm, 1e-2) * (TARGET_SUM / 1e6)
        return (1.0 / n_p + 1.0 / n_c) / mu / (np.log(2.0) ** 2)

    @property
    def signal_var(self) -> np.ndarray:
        """Per-gene variance of the *true* log2FC, as observed variance minus noise.

        Estimated once per table across a sample of perturbations. Doing this per gene
        rather than as one scalar is what keeps a wide, sparsely-expressed gene axis
        usable: Feng's 36,518 genes are mostly near-zero in control, so a single pooled
        signal estimate is dragged under the noise floor and shrinks every gene, informative
        ones included, to zero.
        """
        if getattr(self, "_s2", None) is None:
            rng = np.random.default_rng(0)
            tg = self.targets(min_cells=20)
            sample = list(rng.choice(tg, min(300, len(tg)), replace=False)) if tg else []
            if not sample:
                self._s2 = np.full(len(self.genes), 1e-3)
            else:
                L = np.vstack([self._raw_lfc(g) for g in sample])
                n = np.array([float(self.n_cells[self._row[g]]) for g in sample])
                noise = np.mean([self._noise_var(x) for x in n], axis=0)
                self._s2 = np.maximum(L.var(axis=0) - noise, 1e-4)
        return self._s2

    def _raw_lfc(self, pert: str, pseudocount: float = 1.0) -> np.ndarray:
        v = self.cpm[self._row[pert]].astype(np.float64)
        return np.log2((v + pseudocount) / (self.control_cpm + pseudocount))


def build_tables_multi(path: str, outdir: str, prefix: str, *,
                       group_cols: tuple[str, ...] = ("cell_type", "treatment"),
                       pert_col: str = "condition", chunk: int = 40_000,
                       min_control: int = 200, verbose: bool = True) -> list[dict]:
    """One pass, one table per group -- for files that stack several contexts.

    Jiang's file holds six cell lines crossed with five cytokine treatments in a single
    matrix. Read naively it yields one table whose "control" is an average over all
    thirty contexts, which is not any cell's profile: it transferred at r=0.005, the worst
    of any reference. Splitting it after the fact would mean thirty passes over 25 GB, so
    accumulate every group in the same sweep instead.
    """
    written = []
    with h5py.File(path, "r") as f:
        genes = _var_index(f)
        cats, codes = _codes(f, pert_col)
        dataset = _codes(f, "dataset")[0][0] if "dataset" in f["obs"] else prefix
        n_cells, n_genes = (int(x) for x in f["X"].attrs["shape"])
        labels = []
        for col in group_cols:
            c, k = _codes(f, col)
            labels.append(c[k])
        gkey = np.array(["|".join(x) for x in zip(*labels)])
        groups, gcodes = np.unique(gkey, return_inverse=True)
        acc = np.zeros((len(groups), len(cats), n_genes), dtype=np.float32)
        n_per = np.zeros((len(groups), len(cats)), dtype=np.int64)
        np.add.at(n_per, (gcodes, codes), 1)
        indptr = f["X/indptr"][:]
        for s in range(0, n_cells, chunk):
            e = min(s + chunk, n_cells)
            lo, hi = indptr[s], indptr[e]
            dat = np.expm1(f["X/data"][lo:hi].astype(np.float32))
            idx = f["X/indices"][lo:hi]
            ptr = indptr[s:e + 1] - lo
            rows = np.repeat(np.arange(s, e), np.diff(ptr))
            np.add.at(acc, (gcodes[rows], codes[rows], idx), dat)
            if verbose:
                print(f"    {e:,}/{n_cells:,}", end="\r", flush=True)
    if verbose:
        print()
    ci = list(cats).index(CONTROL) if CONTROL in cats else None
    for g, name in enumerate(groups):
        if ci is None or n_per[g, ci] < min_control:
            if verbose:
                print(f"    [skip] {name}: {0 if ci is None else n_per[g, ci]} control cells")
            continue
        cpm = acc[g].astype(np.float64) / np.maximum(n_per[g], 1)[:, None] * (1e6 / TARGET_SUM)
        slug = name.replace("|", "_").replace("/", "-")
        out = f"{outdir}/{prefix}_{slug}.npz"
        np.savez_compressed(out, cpm=cpm.astype(np.float32), perts=cats, genes=genes,
                            n_cells=n_per[g], cell_line=slug, dataset=str(dataset))
        info = {"out": out, "cell_line": slug, "dataset": str(dataset),
                "n_conditions": int((n_per[g] > 0).sum()), "n_cells": int(n_per[g].sum()),
                "control_cells": int(n_per[g, ci])}
        written.append(info)
        if verbose:
            print(f"    [ok] {slug}: {info['n_conditions']} conditions, "
                  f"{info['n_cells']:,} cells, {info['control_cells']:,} controls")
    return written
