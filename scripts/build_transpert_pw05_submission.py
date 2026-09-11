"""Build a VCC 2026 submission h5ad from one of the public-transfer baseline models.

The scored `transpert_pw05` configuration is:

    python scripts/build_transpert_pw05_submission.py transpert \
        -o out/tp_pw05 --no-shrink --lfc-power 0.5 --target-mean-lfc 0.0898

That configuration scored overall +0.0434 on panel `vcc2026-val-1`
(entry `asT16iuxJgjNkMiaeXU8`). See docs/TRANSPERT_PW05.md.

Requires `data/controls/` (official release) and `data/processed/reftables/`
(public Perturb-seq reference tables built by build_reftables.py). Neither is
committed to this repository.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transpert import emit as E, models as M
from transpert.refdata import RefTable

CTRL_DIR = "data/controls"
REFDIR = "data/processed/reftables"


def load_tables(exclude=()):
    out = []
    for p in sorted(glob.glob(f"{REFDIR}/*.npz")):
        if os.path.basename(p).startswith("_"):
            continue
        try:
            t = RefTable(p)
        except ValueError as e:
            print(f"  [skip] {p}: {e}")
            continue
        if t.cell_line in exclude or t.key in exclude:
            print(f"  [held out] {t.key}")
            continue
        out.append(t)
        print(f"  [ref] {t!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["identity", "mean_response", "transpert", "magnitude", "xlearning"])
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="global response scale")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="weight on the perturbation-specific residual (magnitude model)")
    ap.add_argument("--lfc-power", type=float, default=1.0,
                    help="rank-preserving magnitude compression: sign(x)*|x|**power")
    ap.add_argument("--target-mean-lfc", type=float, default=None,
                    help="rescale the finished prediction to this mean |log2FC|, so "
                         "variants can be compared at matched amplitude")
    ap.add_argument("--confidence", action="store_true",
                    help="weight the specific residual by cross-source agreement")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="compression exponent on the transferred magnitude")
    ap.add_argument("--no-shrink", action="store_true",
                    help="use raw log2FCs instead of the empirical-Bayes shrunk ones")
    ap.add_argument("--temperature", type=float, default=0.05,
                    help="softmax temperature on context similarity (transpert)")
    ap.add_argument("--cells-per-pert", type=int, default=400)
    ap.add_argument("--max-scale", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contexts", default="A,B,C")
    ap.add_argument("--exclude-ref", default="", help="comma-separated cell lines to hold out")
    ap.add_argument("--lfc-npz", default=None,
                    help="precomputed {context}_{pert} log2FC bundle (xlearning)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()

    perts = pd.read_csv(f"{CTRL_DIR}/pert_counts.csv")["target_gene"].tolist()
    genes = pd.read_csv(f"{CTRL_DIR}/gene_names.csv")["gene_name"].to_numpy().astype(str)
    contexts = a.contexts.split(",")
    print(f"{len(perts)} perturbations, {len(genes)} genes, contexts {contexts}")

    tables = []
    if a.model in ("mean_response", "transpert", "magnitude"):
        excl = tuple(x for x in a.exclude_ref.split(",") if x)
        tables = load_tables(exclude=excl)
        if not tables:
            sys.exit("no reference tables — run src/build_reftables.py first")

    lfc_bundle = np.load(a.lfc_npz, allow_pickle=True) if a.lfc_npz else None

    out_h5ad = f"{a.outdir}/prediction.h5ad"
    w = E.SubmissionWriter(out_h5ad, genes)
    meta = {"model": a.model, "alpha": a.alpha, "seed": a.seed,
            "cells_per_pert": a.cells_per_pert, "max_scale": a.max_scale,
            "contexts": contexts, "per_context": {}}

    for ctx in contexts:
        Xc, g, _ = E.load_context(f"{CTRL_DIR}/context_{ctx}.h5ad")
        assert list(g) == list(genes), f"gene order mismatch in context {ctx}"
        ctrl_cpm = E.pseudobulk(Xc) / E.pseudobulk(Xc).sum() * 1e6
        print(f"[{ctx}] controls {Xc.shape} nnz={Xc.nnz:,}  ({time.time()-t0:.0f}s)")

        if a.model == "identity":
            lfc = {}
            info = {}
        elif a.model == "mean_response":
            lfc = M.mean_response(perts, genes, tables, alpha=a.alpha,
                                  shrink=not a.no_shrink)
            info = {"kind": "mean_response"}
        elif a.model == "magnitude":
            lfc, info = M.magnitude_scaled(perts, ctrl_cpm, genes, tables,
                                           alpha=a.alpha, beta=a.beta, gamma=a.gamma,
                                           temperature=a.temperature,
                                           shrink=not a.no_shrink,
                                           confidence_weight=a.confidence,
                                           verbose=True)
        elif a.model == "transpert":
            lfc, info = M.transpert(perts, ctrl_cpm, genes, tables, alpha=a.alpha,
                                    temperature=a.temperature, shrink=not a.no_shrink,
                                    verbose=True)
        else:  # xlearning: predictions were computed by train_xlearning.py
            if lfc_bundle is None:
                sys.exit("--lfc-npz is required for xlearning")
            lfc = {p: a.alpha * lfc_bundle[f"{ctx}|{p}"] for p in perts
                   if f"{ctx}|{p}" in lfc_bundle}
            info = {"kind": "xlearning", "n_covered": len(lfc)}

        if a.lfc_power != 1.0 and lfc:
            lfc = M.compress_lfc(lfc, a.lfc_power)
            info["lfc_power"] = a.lfc_power
        if a.target_mean_lfc is not None and lfc:
            lfc = M.rescale_to(lfc, a.target_mean_lfc)
            info["rescaled_to"] = a.target_mean_lfc

        if lfc:
            mags = np.array([np.abs(v).mean() for v in lfc.values()])
            info["mean_abs_lfc"] = float(mags.mean())
            info["max_abs_lfc"] = float(max(np.abs(v).max() for v in lfc.values()))
        meta["per_context"][ctx] = info

        rng = np.random.default_rng(a.seed + ord(ctx))
        zero = np.zeros(len(genes))
        for i, p in enumerate(perts):
            blk = E.emit(Xc, lfc.get(p, zero), a.cells_per_pert, rng, a.max_scale)
            w.add(blk, p, ctx)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(perts)} perts  nnz={w.nnz:,}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del Xc

    report = w.close()
    report.update(meta)
    report["seconds"] = round(time.time() - t0, 1)
    report["h5ad_bytes"] = os.path.getsize(out_h5ad)
    json.dump(report, open(f"{a.outdir}/report.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
