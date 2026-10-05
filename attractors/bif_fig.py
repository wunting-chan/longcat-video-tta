import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import style
style()
cfgs = [("Native: window 21, no sink", "lobs/main/sf"), ("Window 12, no sink", "lobs/bif_w12_s0/sf"), ("Window 21 + sink 3", "lobs/bif_w21_s3/sf"), ("Window 12 + sink 3", "lobs/bif_w12_s3/sf")]
prompts = [0, 2, 3, 5]; cols = [0, 3, 6, 9]   # thumbnail indices: 0, 60, 120, 180 s
fig, axs = plt.subplots(len(prompts), len(cfgs) * len(cols), figsize=(6.3, 6.3 * len(prompts) / (len(cfgs) * len(cols)) * 120 / 208 * 1.25))
for ci, (name, d) in enumerate(cfgs):
    for ri, p in enumerate(prompts):
        z = np.load(f"{d}/sf_p{p:02d}_s0.npz")
        for k, c in enumerate(cols):
            ax = axs[ri, ci * len(cols) + k]; ax.imshow(z["thumbs"][c]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_linewidth(0.2)
            if ri == 0: ax.set_title(f"{int(z['thumb_idx'][c] / 16)} s", fontsize=4.2, pad=1)
    fig.text((ci + 0.5) / len(cfgs) * 0.98 + 0.01, 0.995, name, ha="center", va="bottom", fontsize=5.6, fontweight="bold")
fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.01, right=0.995, top=0.96, bottom=0.005)
fig.savefig("results_long/bif_strip.png", dpi=220, bbox_inches="tight")
B = json.load(open("results_long/bif.json"))
fig, ax = plt.subplots(1, 3, figsize=(6.3, 1.9))
col = {"w21 s0 (native)": (0, 0, 1), "w12 s0": (0.4, 0.6, 1), "w21 s3": (1, 0, 0), "w12 s3": (1, 0.55, 0.55)}
lab = {"w21 s0 (native)": "w21, no sink (native)", "w12 s0": "w12, no sink", "w21 s3": "w21 + sink 3", "w12 s3": "w12 + sink 3"}
for k, v in B.items():
    ax[0].plot(v["D_prompt"], color=col[k], lw=0.8, label=lab[k]); ax[1].plot(v["sim_to_own_opening"], color=col[k], lw=0.8); ax[2].plot(v["saturation"], color=col[k], lw=0.8)
ax[0].set_title("Distance between different prompts", pad=3); ax[1].set_title("Similarity to own opening", pad=3); ax[2].set_title("Median saturation", pad=3)
for a in ax: a.set_xlabel("Time (s)")
ax[0].legend(fontsize=4.6); fig.tight_layout(w_pad=0.8); fig.savefig("results_long/bif_curves.png", dpi=220, bbox_inches="tight")
T = json.load(open("results_long/twins_dino.json"))
fig, ax = plt.subplots(1, 2, figsize=(6.3, 1.9)); colm = {"sf": (0, 0, 1), "rf": (1, 0, 0), "ll": (0, 0.5, 0)}; nm = {"sf": "Self Forcing", "rf": "Rolling Forcing", "ll": "LongLive"}
tw = json.load(open("results_long/twins.json"))
for m in ("sf", "rf", "ll"):
    for eps, ls in (("0.005", "-"), ("0.05", (0, (3, 1.3)))):
        c = np.array(tw[m][eps]["curve"]); t = (np.arange(len(c)) - 40) * 0.75
        ax[0].semilogy(t[30:], np.maximum(c[30:], 1e-4), color=colm[m], ls=ls, lw=0.8, label=f"{nm[m]} ε={eps}")
        s = np.array(T[f"{m}_{eps}"]); ax[1].plot(np.arange(len(s)) - 30, s, color=colm[m], ls=ls, lw=0.8)
ax[0].axvline(0, color="#777", lw=0.4); ax[1].axvline(0, color="#777", lw=0.4)
ax[0].set_title("Twin divergence, latent space (log)", pad=3); ax[0].set_xlabel("Seconds after perturbation"); ax[0].legend(fontsize=3.8, ncol=2)
ax[1].set_title("Twin divergence, DINOv2 (1 − cos)", pad=3); ax[1].set_xlabel("Seconds after perturbation")
fig.tight_layout(w_pad=1.0); fig.savefig("results_long/twins.png", dpi=220, bbox_inches="tight")
