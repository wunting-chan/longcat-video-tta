"""Sink/window sweep on Self Forcing (seed 0, 8 prompts): does the attractor type change?"""
import glob, os, json, itertools
import numpy as np
FPS = 16
def load(d):
    R = {}
    for f in sorted(glob.glob(f"{d}/*.npz")):
        if "_twin" in f: continue
        z = np.load(f); c = z["cls"].astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
        if int(z["seed"]) != 0: continue
        R[int(z["prompt_index"])] = {"cls": c, "low": z["low"]}
    names = [str(x) for x in z["low_names"]]
    return R, names
def ps(x): n = len(x) // FPS; return x[:n * FPS].reshape(n, FPS, *x.shape[1:]).mean(1)
cfgs = {"w21 s0 (native)": "lobs/main/sf", "w21 s3": "lobs/bif_w21_s3/sf", "w12 s0": "lobs/bif_w12_s0/sf", "w12 s3": "lobs/bif_w12_s3/sf"}
base, names = load(cfgs["w21 s0 (native)"])
att = np.mean([v["cls"][-20 * FPS:].mean(0) for v in base.values()], 0); att /= np.linalg.norm(att)
out = {}
for name, d in cfgs.items():
    R, _ = load(d)
    P = sorted(R); T = min(len(r["cls"]) for r in R.values())
    E = {p: ps(R[p]["cls"][:T]) for p in P}
    for p in P: E[p] /= np.linalg.norm(E[p], axis=1, keepdims=True)
    dprompt = np.mean([1 - (E[a] * E[b]).sum(1) for a, b in itertools.combinations(P, 2)], 0)
    prox = np.mean([E[p] @ att for p in P], 0)                      # closeness to the native SF end state
    own = np.mean([E[p] @ (E[p][:2].mean(0) / np.linalg.norm(E[p][:2].mean(0))) for p in P], 0)
    sat = np.median([ps(np.nan_to_num(R[p]["low"][:T, names.index("saturation")])) for p in P], 0)
    bri = np.median([ps(np.nan_to_num(R[p]["low"][:T, names.index("brightness")])) for p in P], 0)
    out[name] = {"D_prompt": dprompt.tolist(), "prox_to_native_attractor": prox.tolist(), "sim_to_own_opening": own.tolist(),
                 "saturation": sat.tolist(), "brightness": bri.tolist()}
    print(f"{name:16s} different-prompt distance 10s {dprompt[10]:.3f} 90s {dprompt[90]:.3f} end {dprompt[-10:].mean():.3f} | "
          f"sim to own opening end {own[-10:].mean():.3f} | prox to native SF end state end {prox[-10:].mean():.3f} | "
          f"saturation {sat[3:8].mean():.2f}->{sat[-10:].mean():.2f} brightness {bri[3:8].mean():.2f}->{bri[-10:].mean():.2f}")
json.dump(out, open("results_long/bif.json", "w"))
