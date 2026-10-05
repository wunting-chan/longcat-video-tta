import glob, json, numpy as np
FPS = 16; out = {}
def E(f):
    z = np.load(f); c = z["cls"].astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
    n = len(c) // FPS; e = c[:n * FPS].reshape(n, FPS, -1).mean(1); return e / np.linalg.norm(e, axis=1, keepdims=True)
for m in ("sf", "rf", "ll"):
    for eps in ("0.005", "0.05"):
        C = []
        for p in range(8):
            b = f"lobs/main/{m}/{m}_p{p:02d}_s0.npz"; t = f"lobs/main/{m}/{m}_p{p:02d}_s0_twin40e{eps}.npz"
            try: a, c = E(b), E(t)
            except FileNotFoundError: continue
            C.append(1 - (a * c).sum(1))
        C = np.array([x[:min(map(len, C))] for x in C])
        out[f"{m}_{eps}"] = C.mean(0).tolist()
        print(f"{m} eps={eps}: semantic distance at 29s {C[:,29].mean():.4f} | +2s {C[:,32].mean():.3f} +10s {C[:,40].mean():.3f} +60s {C[:,90].mean():.3f} end {C[:,-10:].mean():.3f}")
json.dump(out, open("results_long/twins_dino.json", "w"))
