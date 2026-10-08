"""Free analyses on existing data (no generation), 2026-10-08.
A) Terminal-state structure: cluster the DINOv2 end states (mean of the last 10 s) of every Self Forcing
   rollout we have (main 180 s, Stage A arms 120 s, structure-only 180 s, kicks, swaps), plus RF/LL main runs.
   Silhouette over k = 2..8 tells one basin vs a few discrete basins vs a continuum.
B) VBench vs semantics: per video and 10-s window (10-20, 55-65, 110-120 s), correlate VBench dimensions with
   (i) prompt identification in that window (vs the other seed of the same arm) and (ii) the arm's cross-prompt
   diversity in that window, pooled over Stage A arms (Spearman, 95% bootstrap CI over videos).
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

FPS = 16
R = "."


def per_second(f):
    d = np.load(f)
    c = d["cls"].astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
    n = len(c) // FPS
    z = c[: n * FPS].reshape(n, FPS, -1).mean(1)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


# ---------------------------------------------------------------- A
ends, tags = [], []
for f in sorted(glob.glob(f"{R}/lobs/*/*/*.npz")):
    group = f.split("/lobs/")[1].split("/")[0]
    if group in ("vb_test",):
        continue
    stem = os.path.basename(f)[:-4]
    if "_twin" in stem:
        continue
    z = per_second(f)
    if len(z) < 100:
        continue
    e = z[-10:].mean(0); e /= np.linalg.norm(e)
    ends.append(e); tags.append((group, stem))
E = np.array(ends)
model = np.array([t[1].split("_")[0] for t in tags])
from scipy.cluster.vq import kmeans2


def kmeans(X, k, n_init=10, seed=0):
    best, bl = None, None
    for i in range(n_init):
        c, lab = kmeans2(X, k, minit="++", seed=seed + i)
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best:
            best, bl = inertia, lab
    return bl


def silhouette(X, lab):
    D = 1 - X @ X.T                                   # cosine distance (rows are unit norm)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == l].mean() for l in np.unique(lab) if l != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())
res = {"n_end_states": len(E), "by_model": {m: int((model == m).sum()) for m in set(model)}}
for name, mask in (("all", np.ones(len(E), bool)), ("sf_only", model == "sf")):
    X = E[mask]; sil = {}
    for k in range(2, 9):
        lab = kmeans(X, k)
        sil[k] = silhouette(X, lab)
    kbest = max(sil, key=sil.get)
    lab = kmeans(X, kbest)
    sizes = np.bincount(lab).tolist()
    res[name] = {"silhouette": sil, "k_best": kbest, "sizes": sizes,
                 "members": defaultdict(list)}
    for i, (t, l) in enumerate(zip(np.array(tags, dtype=object)[mask], lab)):
        arm = t[0] + ":" + "_".join(t[1].split("_")[3:]) if len(t[1].split("_")) > 3 else t[0] + ":base"
        res[name]["members"][int(l)].append(arm)
    res[name]["members"] = {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in res[name]["members"].items()}
    res[name]["members"] = {k: {a: int(c) for a, c in v.items()} for k, v in res[name]["members"].items()}
    print(f"A) {name}: n={len(X)}  silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, " best k =", kbest, "sizes", sizes)
# pairwise end-state distance distribution SF vs RF vs LL main runs (cross prompt)
for m in ("sf", "rf", "ll"):
    idx = [i for i, t in enumerate(tags) if t[0] == "main" and model[i] == m]
    if len(idx) > 2:
        D = 1 - E[idx] @ E[idx].T
        print(f"   main {m}: mean end-state distance {D[np.triu_indices(len(idx), 1)].mean():.3f}")

# ---------------------------------------------------------------- B
vb = defaultdict(dict)
for r in csv.DictReader(open(f"{R}/vbench/stageA/vbench_segments.csv")):
    s = r["score"]; vb[(r["video"], r["window"])][r["dim"]] = (1.0 if s == "True" else 0.0) if s in ("True", "False") else float(s)
ARMS = {"": "baseline", "_repel1_mem": "mem", "_repel1_mem_w2.25-60": "mem_early", "_random1_mem": "mem_rand",
        "_repel1": "out", "_bon4score": "bo4s", "_bon4random": "bo4r"}
W = {"w010": (10, 20), "w055": (55, 65), "w110": (110, 120)}
rows = []
for suf, arm in ARMS.items():
    Z = {(p, s): per_second(f"{R}/lobs/stageA/sf/sf_p{p:02d}_s{s}{suf}.npz") for p in range(8, 16) for s in (0, 1)}
    P = list(range(8, 16))
    for w, (a, b) in W.items():
        Zw = {k: v[a:b].mean(0) / np.linalg.norm(v[a:b].mean(0)) for k, v in Z.items()}
        X0 = np.stack([Zw[(p, 0)] for p in P]); X1 = np.stack([Zw[(p, 1)] for p in P])
        C = X0 @ X1.T
        div = float((1 - C[~np.eye(8, dtype=bool)]).mean())
        for i, p in enumerate(P):
            for s, hit in ((0, float(C[i].argmax() == i)), (1, float(C[:, i].argmax() == i))):
                v = vb.get((f"sf_p{p:02d}_s{s}{suf}", w), {})
                rows.append({"arm": arm, "window": w, "ident": hit, "diversity": div, **v})
dims = ["imaging_quality", "aesthetic_quality", "temporal_flickering", "subject_consistency", "dynamic_degree"]
out_b = {}
print("B) Spearman correlation, VBench dimension vs semantic measures (pooled: 7 arms x 16 videos x 3 windows)")
rng = np.random.default_rng(0)
for d in dims:
    x = np.array([r.get(d, np.nan) for r in rows]); ok = ~np.isnan(x)
    for tgt in ("ident", "diversity"):
        y = np.array([r[tgt] for r in rows])
        rho = spearmanr(x[ok], y[ok])[0]
        bs = []
        for _ in range(1000):
            j = rng.choice(np.where(ok)[0], ok.sum()); bs.append(spearmanr(x[j], y[j])[0])
        out_b[f"{d}~{tgt}"] = [float(rho), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"   {d:22s} ~ {tgt:9s}: rho {rho:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
# time course of means per window (baseline arm)
tc = {}
for w in W:
    rr = [r for r in rows if r["window"] == w]
    tc[w] = {k: float(np.nanmean([r.get(k, np.nan) for r in rr])) for k in dims + ["ident", "diversity"]}
print("   means by window (all arms):", {w: {k: round(v, 3) for k, v in d.items()} for w, d in tc.items()})
json.dump({"A": res, "B_corr": out_b, "B_timecourse": tc}, open(f"{R}/results_long/free_round2.json", "w"), indent=1, default=str)

# within-window correlations (removes the shared time trend); ident is per video, diversity is per arm (n = 7 per window)
print("B2) within-window Spearman (per-video ident; per-arm diversity uses arm means)")
out_w = {}
for w in W:
    rr = [r for r in rows if r["window"] == w]
    for d in dims:
        x = np.array([r.get(d, np.nan) for r in rr]); y = np.array([r["ident"] for r in rr]); ok = ~np.isnan(x)
        rho_i = spearmanr(x[ok], y[ok])[0] if y[ok].std() > 0 else float("nan")
        arms = sorted(set(r["arm"] for r in rr))
        xa = [np.nanmean([r.get(d, np.nan) for r in rr if r["arm"] == a]) for a in arms]
        ya = [next(r["diversity"] for r in rr if r["arm"] == a) for a in arms]
        rho_d = spearmanr(xa, ya)[0]
        out_w[f"{w}:{d}"] = [float(rho_i), float(rho_d)]
        print(f"   {w} {d:22s}: ~ident rho {rho_i:+.2f} (n={ok.sum()})   ~arm diversity rho {rho_d:+.2f} (n={len(arms)})")
json.dump(out_w, open(f"{R}/results_long/free_round2_within.json", "w"), indent=1)
