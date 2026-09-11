"""The baseline zoo. Every model returns one log2 fold-change vector per perturbation.

Emission is shared (``emit.build_submission``), so a model is exactly a map
``target gene -> log2FC over the challenge gene axis``. That factoring is not a
simplification for convenience: five of the six vcc2026 metrics read only the
per-perturbation pseudobulk and the DE call on top of it, and the sixth (``pds_cosine``)
ranks perturbations by their pseudobulk delta. Cell-level structure enters only through
the Wilcoxon test's variance, which the resample-and-scale emission inherits from the
real control cells.

⚠️ The scorer excludes each perturbation's own target gene from all six metrics
(``exclude_target_gene: true``, ``exclusion_scope: panel`` -- in fact all 300 panel genes
are dropped from ``pds_cosine``). Predicting the knockdown itself scores nothing. The
entire signal is the downstream response.

Models
------
``identity``       zero response; where "do nothing" lands.
``mean_response``  one average response applied to every perturbation. The shape of
                   cell-eval2's own context-mean baseline, which defines the 0 point --
                   except estimated from external cell lines rather than the held-out
                   truth, so it will not land exactly at 0.
``transpert``      after Team Outlier (VCC-2025, 3rd): transfer each target's measured
                   log2FC from the reference cell lines that carry it, weighted by how
                   similar each reference's control profile is to this context, then
                   scaled globally.
``xlearning``      after Team XLearning Lab (VCC-2025, 2nd): an MLP on aggregated control
                   expression + the target's ESM-2 protein embedding, regressing the
                   log2FC residual under MSE.
"""
from __future__ import annotations

import numpy as np

CONTROL = "non-targeting"


# --------------------------------------------------------------------------- helpers
_ALIGN_CACHE: dict = {}


def align_index(src_genes: np.ndarray, dst_genes: np.ndarray):
    """Cached (take, hit) index pair mapping one gene axis onto another.

    Rebuilding the symbol map per call costs a 36,000-entry dict and a Python loop every
    time a profile moves axes -- which, in a leave-one-out sweep, is once per
    (target, source) pair. The axes are fixed per table, so key on identity.
    """
    # The entry keeps a reference to BOTH axes and re-checks them with `is`. Keying on
    # id() alone is a live bug here: np.intersect1d hands back a temporary, CPython
    # recycles its address once it is freed, and the next temporary at the same address
    # silently collects the previous array's index -- a wrong-length gather, or worse, a
    # right-length one that is quietly misaligned.
    key = (id(src_genes), id(dst_genes))
    e = _ALIGN_CACHE.get(key)
    if e is not None and e[0] is src_genes and e[1] is dst_genes:
        return e[2], e[3]
    pos = {g: i for i, g in enumerate(src_genes)}
    take = np.fromiter((pos.get(g, -1) for g in dst_genes), dtype=np.int64,
                       count=len(dst_genes))
    hit = take >= 0
    if len(_ALIGN_CACHE) > 256:          # bounded: the hot pairs are long-lived attributes
        _ALIGN_CACHE.clear()
    _ALIGN_CACHE[key] = (src_genes, dst_genes, take, hit)
    return take, hit


def align(values: np.ndarray, src_genes: np.ndarray, dst_genes: np.ndarray,
          fill: float = 0.0) -> np.ndarray:
    """Move a per-gene vector onto another gene axis by symbol; unmatched -> ``fill``."""
    take, hit = align_index(src_genes, dst_genes)
    out = np.full(len(dst_genes), fill, dtype=np.float64)
    out[hit] = np.asarray(values, dtype=np.float64)[take[hit]]
    return out


def log_cpm(cpm: np.ndarray) -> np.ndarray:
    return np.log2(np.asarray(cpm, dtype=np.float64) + 1.0)


def context_similarity(ctx_cpm: np.ndarray, ctx_genes: np.ndarray, table) -> float:
    """Pearson r between a reference cell line's control profile and this context's.

    Computed on log2(CPM+1) over the genes the two share, which is the only handle we
    have on "is this reference cell line like the context I have to predict" -- the
    challenge never names the cell lines.
    """
    shared = np.intersect1d(ctx_genes, table.genes)
    if len(shared) < 500:
        return 0.0
    a = log_cpm(align(ctx_cpm, ctx_genes, shared))
    b = log_cpm(align(table.control_cpm, table.genes, shared))
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- models
def identity(perts: list[str], n_genes: int) -> dict[str, np.ndarray]:
    return {}  # emit() treats a missing perturbation as zero log2FC


def mean_response(perts, ctx_genes, tables, alpha: float = 1.0,
                  min_cells: int = 50, shrink: bool = True) -> dict[str, np.ndarray]:
    """One response vector, the across-perturbation mean, given to every perturbation."""
    acc = np.zeros(len(ctx_genes))
    n = 0
    for t in tables:
        for g in t.targets(min_cells=min_cells):
            acc += align(t.lfc(g, shrink=shrink), t.genes, ctx_genes)
            n += 1
    if n == 0:
        return {}
    v = alpha * acc / n
    return {p: v for p in perts}


def transpert(perts, ctx_cpm, ctx_genes, tables, *, alpha: float = 1.0,
              temperature: float = 0.05, min_cells: int = 20,
              shrink_cells: float = 50.0, fallback_mean: bool = True,
              shrink: bool = True, verbose: bool = False):
    """Similarity-weighted transfer of each target's measured response.

    For target ``g`` we take every reference cell line that knocked ``g`` down and average
    their log2FC vectors, weighting each by ``softmax(r / temperature)`` on the control-
    profile similarity ``r`` and by a shrinkage factor ``n/(n + shrink_cells)`` that pulls
    a reference measured on a handful of cells toward zero. ``alpha`` scales the result:
    a transferred response is systematically too large for a context where the gene
    matters less, and PDS in particular is sensitive to the scale of the predicted effect
    (Liu et al., arXiv:2511.16954).

    Targets no reference carries fall back to the mean response, which is the honest
    answer for "some perturbation happened, we do not know which".
    """
    sims = {t.key: context_similarity(ctx_cpm, ctx_genes, t) for t in tables}
    if verbose:
        for k, v in sorted(sims.items(), key=lambda kv: -kv[1]):
            print(f"    similarity {k:<24} r={v:+.3f}")
    have = {}
    for t in tables:
        for g in t.targets(min_cells=min_cells):
            have.setdefault(g, []).append(t)

    fb = np.zeros(len(ctx_genes))
    if fallback_mean:
        mr = mean_response(["_"], ctx_genes, tables, alpha=1.0, min_cells=min_cells,
                           shrink=shrink)
        fb = mr.get("_", fb)

    out, n_hit = {}, 0
    for p in perts:
        srcs = have.get(p, [])
        if not srcs:
            out[p] = alpha * fb
            continue
        n_hit += 1
        w = np.array([np.exp(sims[t.key] / temperature) for t in srcs])
        n_c = np.array([float(t.n_cells[list(t.perts).index(p)]) for t in srcs])
        w = w * (n_c / (n_c + shrink_cells))
        w = w / w.sum() if w.sum() > 0 else np.full(len(srcs), 1 / len(srcs))
        acc = np.zeros(len(ctx_genes))
        for wi, t in zip(w, srcs):
            acc += wi * align(t.lfc(p, shrink=shrink), t.genes, ctx_genes)
        out[p] = alpha * acc
    if verbose:
        print(f"    {n_hit}/{len(perts)} targets found in reference tables")
    return out, {"n_covered": n_hit, "n_perts": len(perts), "similarities": sims}


# --------------------------------------------------------------------------- xlearning
class XLearningMLP:
    """The 2nd-place shape: control expression + ESM-2 target embedding -> log2FC.

    Team XLearning Lab described "a Fully Connected Network using aggregated control
    expression, ESM-2 protein embeddings for perturbed genes, and UMI count indicators"
    with residual learning under MSE on pseudobulk. Regressing log2FC *is* the residual
    parameterization: the identity prediction is the zero vector, so the network only ever
    has to explain the departure from control.

    The control profile enters through a PCA basis fit across the reference cell lines --
    18,533 raw inputs against a few thousand training rows is a memorization machine, and
    what the network needs from the context is "which kind of cell is this", which the
    leading components carry.
    """

    def __init__(self, n_out: int, esm_dim: int, ctx_dim: int = 128,
                 hidden: int = 1024, depth: int = 2, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        d_in = esm_dim + ctx_dim + 1
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
                       nn.Dropout(0.1)]
            d = hidden
        layers += [nn.Linear(d, n_out)]
        # start at the identity prediction: zero log2FC everywhere
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
        self.ctx_dim = ctx_dim
        self.torch = torch

    def fit(self, X_esm, X_ctx, X_umi, Y, *, mask=None, epochs=200, lr=1e-3, batch=64,
            weight_decay=1e-4, val=None, verbose=True):
        torch = self.torch
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xi = torch.tensor(np.hstack([X_esm, X_ctx, X_umi[:, None]]), dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        # every source measures a different gene subset; scoring the loss on genes it
        # never saw would train the network to call them unresponsive
        Mt = (torch.tensor(mask, dtype=torch.float32) if mask is not None
              else torch.ones_like(Yt))
        n = len(Xi)
        hist = []
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                opt.zero_grad()
                d = (self.net(Xi[b]) - Yt[b]) * Mt[b]
                loss = (d ** 2).sum() / Mt[b].sum().clamp(min=1.0)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            sched.step()
            row = {"epoch": ep, "train_mse": tot / n}
            if val is not None and (ep % 10 == 0 or ep == epochs - 1):
                row["val_mse"] = self.score(*val)
            hist.append(row)
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print("    " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else
                                         f"{k}={v}" for k, v in row.items()), flush=True)
        return hist

    def predict(self, X_esm, X_ctx, X_umi) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        Xi = torch.tensor(np.hstack([X_esm, X_ctx, X_umi[:, None]]), dtype=torch.float32)
        with torch.no_grad():
            return self.net(Xi).numpy().astype(np.float64)

    def score(self, X_esm, X_ctx, X_umi, Y, mask=None) -> float:
        d = self.predict(X_esm, X_ctx, X_umi) - Y
        if mask is None:
            return float((d ** 2).mean())
        return float((d ** 2 * mask).sum() / max(mask.sum(), 1.0))


# --------------------------------------------------------------------- magnitude-scaled
def magnitude_scaled(perts, ctx_cpm, ctx_genes, tables, *, alpha: float = 1.0,
                     beta: float = 1.0, gamma: float = 1.0, temperature: float = 0.05,
                     min_cells: int = 20, shrink: bool = True,
                     confidence_weight: bool = False, verbose: bool = False):
    """Generic response scaled per perturbation, plus a weighted specific residual.

    Measured across every pair of reference cell lines (``src/diag_transfer_magnitude.py``):
    a perturbation's response **magnitude** transfers at Spearman 0.45 median (0.53-0.75 on
    well-powered pairs, 0.60 for K562 -> H1), while its per-gene **shape** transfers at
    0.064. Size is a single number estimated from hundreds of cells; shape is 18,000
    numbers each estimated from the same cells. So the size is the part worth carrying.

    That gap is worth exploiting because it is exactly what the context-mean baseline
    throws away. The baseline hands every perturbation one identical response, so every
    perturbation gets the same number of significant genes -- while real yields span orders
    of magnitude. ``de_wilcoxon_direction_fidelity_yield_raw`` is
    ``k / max(n_pred, n_real)`` and ``de_wilcoxon_sig_jaccard`` is a set overlap: both are
    dominated by getting the *count* right before the identity of the genes matters.

        lfc[p] = alpha * ( m_p * generic  +  beta * (spec[p] - mean_p spec[p]) )

    ``m_p`` is the transferred magnitude, normalized to mean 1 so ``alpha`` alone carries
    the overall scale; ``generic`` is the across-perturbation mean response (the baseline's
    own direction); the residual is what similarity-weighted transfer adds beyond it.
    ``beta=0`` is the pure magnitude model, ``beta=1`` adds the full specific signal.
    """
    sims = {t.key: context_similarity(ctx_cpm, ctx_genes, t) for t in tables}
    have: dict[str, list] = {}
    for t in tables:
        for g in t.targets(min_cells=min_cells):
            have.setdefault(g, []).append(t)

    generic = mean_response(["_"], ctx_genes, tables, alpha=1.0, min_cells=min_cells,
                            shrink=shrink)["_"]

    spec, mag, conf = {}, {}, {}
    for p in perts:
        srcs = have.get(p, [])
        if not srcs:
            spec[p] = np.zeros(len(ctx_genes))
            mag[p] = np.nan          # filled with the panel median below
            conf[p] = 0.0            # nothing carries it: no specific signal to trust
            continue
        w = np.array([np.exp(sims[t.key] / temperature) for t in srcs])
        n_c = np.array([float(t.n_cells[t._row[p]]) for t in srcs])
        w = w * (n_c / (n_c + 50.0))
        w = w / w.sum() if w.sum() > 0 else np.full(len(srcs), 1 / len(srcs))
        acc = np.zeros(len(ctx_genes))
        m = 0.0
        for wi, t in zip(w, srcs):
            v = t.lfc(p, shrink=shrink)
            acc += wi * align(v, t.genes, ctx_genes)
            # magnitude on the reference's OWN axis: a narrow HVG panel would otherwise
            # read as a bigger response purely for having dropped its quiet genes
            m += wi * float(np.abs(v).mean())
        spec[p] = acc
        mag[p] = m
        # Confidence = do independent references AGREE about this target's response?
        # Two cell lines that saw the same knockdown and produced correlated profiles
        # are evidence the signal is real; one thin arm, or two that disagree, is not.
        # 145 of the 300 2026 targets are carried by two or more references.
        if len(srcs) >= 2:
            prof = []
            for t in srcs:
                v = align(t.lfc(p, shrink=shrink), t.genes, ctx_genes)
                v = v - v.mean()
                nv = np.linalg.norm(v)
                prof.append(v / nv if nv > 0 else v)
            cors = [float(prof[i] @ prof[j])
                    for i in range(len(prof)) for j in range(i + 1, len(prof))]
            conf[p] = float(np.clip(np.mean(cors), 0.0, 1.0))
        else:
            conf[p] = np.nan          # filled with the panel median below

    vals = np.array([mag[p] for p in perts], dtype=float)
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 1.0
    vals = np.where(np.isfinite(vals), vals, med)
    # Normalise by the MEDIAN, not the mean. The transferred magnitudes are strongly
    # right-skewed (on the H1 panel: median 0.23 against mean 1.0), so mean-1 scaling
    # pushes more than half the panel below 1x, costing DE yield on exactly the
    # coverage term -- k/max(n_pred, n_real) -- the model was built to win. Measured:
    # mean-normalised scored fid -0.210 against the flat baseline's -0.110.
    # gamma < 1 compresses the tail: the ranking of magnitudes transfers far better
    # than their ratios do.
    vals = vals / max(med, 1e-12)
    if gamma != 1.0:
        vals = np.power(np.maximum(vals, 1e-6), gamma)
    mean_spec = np.mean([spec[p] for p in perts], axis=0)

    cvals = np.array([conf[p] for p in perts], dtype=float)
    if confidence_weight:
        med_c = float(np.nanmedian(cvals)) if np.isfinite(cvals).any() else 0.5
        cvals = np.where(np.isfinite(cvals), cvals, med_c)
        # normalise to mean 1 so beta keeps its meaning: this reweights across the panel,
        # it does not quietly rescale the whole specific term
        w = cvals / max(cvals.mean(), 1e-12)
    else:
        w = np.ones(len(perts))

    out = {}
    for p, m, wp in zip(perts, vals, w):
        out[p] = alpha * (m * generic + beta * wp * (spec[p] - mean_spec))
    if verbose:
        print(f"    magnitude spread (median-normalised, gamma={gamma}): "
              f"p10={np.quantile(vals, .1):.2f} median={np.median(vals):.2f} "
              f"p90={np.quantile(vals, .9):.2f} max={vals.max():.2f}")
        print(f"    {sum(1 for p in perts if p in have)}/{len(perts)} targets in reference tables")
        if confidence_weight:
            print(f"    cross-source agreement: p10={np.quantile(w, .1):.2f} "
                  f"median={np.median(w):.2f} p90={np.quantile(w, .9):.2f}  "
                  f"({sum(1 for p in perts if len(have.get(p, [])) >= 2)} targets with >=2 sources)")
    return out, {"n_covered": sum(1 for p in perts if p in have), "n_perts": len(perts),
                 "magnitude_p10_p90": [float(np.quantile(vals, .1)),
                                       float(np.quantile(vals, .9))],
                 "similarities": sims}


def rescale_to(lfc: dict[str, np.ndarray], target_mean_abs: float) -> dict[str, np.ndarray]:
    """Rescale a whole prediction so its mean |log2FC| equals ``target_mean_abs``.

    Every reweighting tried so far -- per-perturbation magnitude, cross-source agreement --
    changed the prediction's amplitude as a side effect, and amplitude turned out to
    dominate the score: on RPE1 the confidence-weighted model moved mean|lfc| 0.099 -> 0.138
    and `nmae` fell +0.038 -> -0.387, burying whatever the reweighting did to the shape.
    Comparing variants at matched amplitude separates the two questions, so a shape change
    can be judged on its own and the scale tuned once, separately.
    """
    cur = float(np.mean([np.abs(v).mean() for v in lfc.values()])) if lfc else 0.0
    if cur <= 0:
        return lfc
    k = target_mean_abs / cur
    return {p: v * k for p, v in lfc.items()}


def compress_lfc(lfc: dict[str, np.ndarray], power: float) -> dict[str, np.ndarray]:
    """Rank-preserving compression of the log2FC magnitude distribution.

    ``nmae``'s baseline is NMAE ~= 1.0, which is what predicting *zero* scores. The first
    leaderboard entry returned −0.207, i.e. the per-gene fold changes we supply on the
    reference-significant genes are worse than supplying none -- while ``pds`` was +0.211,
    so the ordering of the response carries real information. That combination points at
    the magnitude distribution rather than the ranking: a transferred profile inherits the
    source line's dynamic range, and its loud tail is where absolute error accumulates.

    ``sign(x) * |x|**power`` is monotone, so it leaves every gene's rank within a
    perturbation untouched -- what ``pds`` reads is unchanged -- while pulling the tail in
    relative to the middle. Callers should follow with ``rescale_to`` so the comparison is
    at matched amplitude and only the distribution's shape has moved.
    """
    if power == 1.0:
        return lfc
    return {p: np.sign(v) * np.power(np.abs(v), power) for p, v in lfc.items()}
