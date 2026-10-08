"""State transplant and prompt switch, 2026-10-08. Held-out prompts 8-15, seed 0, 120 s.
Donor/new prompt of recipient p is 8 + ((p - 8 + 1) % 8). All prompts at one seed share input noise,
so the recipient, the donor and every arm see identical noise.

Per second t (DINOv2 [CLS], unit norm, averaged per second):
  content(t) = cos(V_t, open_donor) - cos(V_t, open_own)   opening centroid = mean of baseline 0-20 s
               > 0 means the frames look more like the donor/new prompt's opening content than their own.
  d_attr(t)  = 1 - cos(V_t, a)   a = frozen SF attractor estimate from prompts 0-7 (held out here).
Reported in windows relative to the intervention time, means over the 8 prompts with a bootstrap 95% CI.
"""
import json
import os

import numpy as np

FPS = 16
R = "."
P = list(range(8, 16))
donor = {p: 8 + ((p - 8 + 1) % 8) for p in P}
rng = np.random.default_rng(0)


def per_second(f):
    d = np.load(f)
    c = d["cls"].astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
    n = len(c) // FPS
    z = c[: n * FPS].reshape(n, FPS, -1).mean(1)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def ci(x):
    x = np.asarray(x, float)
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


att = np.load(f"{R}/dino_attractor_sf_p0-7.npy").astype(np.float32).ravel(); att /= np.linalg.norm(att)
BASE = {"sf": lambda p: f"{R}/lobs/stageA/sf/sf_p{p:02d}_s0.npz",
        "ll": lambda p: f"{R}/lobs/mech/ll/ll_p{p:02d}_s0.npz"}
ARMS = {"sf": [("tpkv30", "_tpkv30", 30), ("tplat30", "_tplatest30", 30),
               ("sw10", "_sw10", 10), ("sw30", "_sw30", 30), ("sw60", "_sw60", 60), ("sw90", "_sw90", 90)],
        "ll": [("sw10", "_sw10", 10), ("sw30", "_sw30", 30), ("sw60", "_sw60", 60), ("sw90", "_sw90", 90)]}
out = {}
for m in ("sf", "ll"):
    if not all(os.path.exists(BASE[m](p)) for p in P):
        print(m, "baselines missing; skipped"); continue
    B = {p: per_second(BASE[m](p)) for p in P}
    O = {p: B[p][:20].mean(0) / np.linalg.norm(B[p][:20].mean(0)) for p in P}
    # reference: the donor's own baseline, content index relative to the recipient
    for name, suf, at in [("baseline", None, None)] + ARMS[m]:
        if suf is None:
            V = B
        else:
            fs = {p: f"{R}/lobs/mech/{m}/{m}_p{p:02d}_s0{suf}.npz" for p in P}
            if not all(os.path.exists(f) for f in fs.values()):
                print(m, name, "missing", sum(not os.path.exists(f) for f in fs.values())); continue
            V = {p: per_second(fs[p]) for p in P}
        T = min(len(v) for v in V.values())
        content = np.stack([V[p][:T] @ O[donor[p]] - V[p][:T] @ O[p] for p in P])
        dattr = np.stack([1 - V[p][:T] @ att for p in P])
        own_track = np.stack([np.sum(V[p][:T] * B[p][:T], 1) for p in P])      # cos to own baseline at same t
        don_track = np.stack([np.sum(V[p][:T] * B[donor[p]][:T], 1) for p in P])  # cos to donor baseline at same t
        r = {"curve_content": content.mean(0).round(4).tolist(), "curve_dattr": dattr.mean(0).round(4).tolist()}
        ref = at if at is not None else 0
        for lab, (a, b) in {"pre": (ref - 5, ref), "+0-10": (ref, ref + 10), "+10-30": (ref + 10, ref + 30),
                            "end110-120": (110, 120)}.items():
            if a < 0 or b > T:
                continue
            r[lab] = {"content": ci(content[:, a:b].mean(1)), "d_attr": ci(dattr[:, a:b].mean(1)),
                      "cos_own_baseline": ci(own_track[:, a:b].mean(1)),
                      "cos_donor_baseline": ci(don_track[:, a:b].mean(1)),
                      "closer_to_donor_frac": float((don_track[:, a:b].mean(1) > own_track[:, a:b].mean(1)).mean())}
        out[f"{m}:{name}"] = r
        msg = "  ".join(f"{k}: content {v['content'][0]:+.3f} d_attr {v['d_attr'][0]:.3f} donor-closer {v['closer_to_donor_frac']:.2f}"
                        for k, v in r.items() if not k.startswith("curve"))
        print(f"{m} {name:9s} {msg}")
# baseline reference for the switch: what the donor prompt itself looks like (content index of donor baseline vs recipient)
for m in ("sf", "ll"):
    if all(os.path.exists(BASE[m](p)) for p in P):
        B = {p: per_second(BASE[m](p)) for p in P}
        O = {p: B[p][:20].mean(0) / np.linalg.norm(B[p][:20].mean(0)) for p in P}
        c = np.stack([B[donor[p]][:120] @ O[donor[p]] - B[donor[p]][:120] @ O[p] for p in P])
        out[f"{m}:donor_ceiling"] = {"curve_content": c.mean(0).round(4).tolist()}
        print(f"{m} donor's own video, content index by window: " +
              "  ".join(f"{a}-{b}s {c[:, a:b].mean():+.3f}" for a, b in ((0, 10), (20, 40), (50, 70), (80, 100), (110, 120))))
os.makedirs(f"{R}/results_long", exist_ok=True)
json.dump(out, open(f"{R}/results_long/mech_analysis.json", "w"), indent=1)
