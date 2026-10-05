"""Twin divergence after a noise perturbation at block 40 (30 s), for perturbation sizes 0.005 and 0.05.
Latent space: saved every 3rd latent (one per 3-latent block), exact. Pixel-semantic space: DINOv2 per second."""
import glob, json, os, re
import numpy as np, torch
L = "lat"  # local copies of *.lat3.pt
FPS = 16; out = {}
def latents(model, stem):
    return torch.load(f"{L}/{model}/{stem}.lat3.pt").float().numpy()
for model in ("sf", "rf", "ll"):
    res = {}
    for eps in ("0.005", "0.05"):
        curves = []
        for p in range(8):
            base = f"{model}_p{p:02d}_s0"; twin = f"{base}_twin40e{eps}"
            if not os.path.exists(f"{L}/{model}/{twin}.lat3.pt"):
                continue
            a, b = latents(model, base), latents(model, twin)
            d = np.linalg.norm((a - b).reshape(len(a), -1), axis=1) / np.linalg.norm(a.reshape(len(a), -1), axis=1)
            curves.append(d)
        if not curves:
            continue
        C = np.array(curves)                       # [prompt, block]; block 40 is the perturbed one
        pre = C[:, :40].max()
        g = np.log(np.maximum(C[:, 40:], 1e-6))
        # growth rate: slope of mean log distance over the first 10 blocks after the perturbation (1 block = 3 latents = 0.75 s)
        lam = np.polyfit(np.arange(10) * 0.75, g[:, :10].mean(0), 1)[0]
        res[eps] = {"n": len(C), "max_before": float(pre), "d_at_perturb": float(C[:, 40].mean()),
                    "d_plus_5s": float(C[:, 40 + 7].mean()), "d_plus_30s": float(C[:, 40 + 40].mean()),
                    "d_end": float(C[:, -10:].mean()), "lambda_per_s_first_7.5s": float(lam), "curve": C.mean(0).tolist()}
    out[model] = res
    for eps, r in res.items():
        print(f"{model} eps={eps}: n={r['n']} pre-max {r['max_before']:.1e} | d@perturb {r['d_at_perturb']:.3f} +5s {r['d_plus_5s']:.3f} +30s {r['d_plus_30s']:.3f} end {r['d_end']:.3f} | growth {r['lambda_per_s_first_7.5s']:+.3f}/s")
json.dump(out, open("results_long/twins.json", "w"), indent=1)
