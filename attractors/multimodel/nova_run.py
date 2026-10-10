#!/usr/bin/env python3
"""NOVA text-to-video (BAAI/nova-d48w1024-osp480) with the README call, length set by latents.

The README example uses max_latent_length=9 (33 frames at 12 fps). Longer videos just raise
max_latent_length; whether the model accepts it is what the Phase A probe checks.
"""
import argparse, json, os, time
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_video

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--latents", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
prompt = open(a.prompt_file).read().strip()
pipe = NOVAPipeline.from_pretrained("BAAI/nova-d48w1024-osp480", torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
g = torch.Generator("cuda").manual_seed(a.seed)
t0 = time.time()
video = pipe(prompt, max_latent_length=a.latents, generator=g).frames[0]
dt = time.time() - t0
export_to_video(video, a.out, fps=12)
print(json.dumps({"frames": len(video), "fps": 12, "gen_seconds": dt}), flush=True)
