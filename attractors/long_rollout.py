#!/usr/bin/env python3
"""Phase 0 of the attractor study: multi-minute T2V rollouts from three streaming students on
Wan2.1-T2V-1.3B, using each repo's own inference pipeline unmodified.

  sf  Self Forcing (DMD, EMA).    local_attn_size=21 turns on its rolling KV window; the
                                  attention span is the same 21 latents it uses natively.
  rf  Rolling Forcing (DMD, EMA). Native rolling window with a 24-latent cache and a sink.
  ll  LongLive (base + LoRA).     Its 'infinity' settings: local_attn 12, sink 3.

Seeding: for each (prompt, seed) the noise tensor is drawn right after torch.manual_seed(seed),
so every prompt at a given seed receives the identical noise (common-noise synchronization test).
Third-party repos are imported read-only; nothing in them is modified. Writes only under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

TP = "/scratch/wc3013/third_party"
CK = "/scratch/wc3013/wan-checkpoints"
REPO = {"sf": f"{TP}/Self-Forcing", "rf": f"{TP}/RollingForcing", "ll": f"{TP}/LongLive"}


def sdpa_fallback():
    """Same patch the TTA runners use (run_i2v_continuation.install_sdpa_attention_fallback):
    flash-attn is not installed in the self_forcing env, and Wan's model.py calls flash_attention
    by name. Route it to the SDPA path of wan.modules.attention.attention()."""
    from wan.modules import attention as attn
    from wan.modules import model as wan_model
    if getattr(attn, "FLASH_ATTN_2_AVAILABLE", False) or getattr(attn, "FLASH_ATTN_3_AVAILABLE", False):
        print("flash-attn available; no SDPA fallback", flush=True)
        return

    def _sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
        return attn.attention(q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p, softmax_scale=softmax_scale,
                              q_scale=q_scale, causal=causal, window_size=window_size, deterministic=deterministic,
                              dtype=dtype, fa_version=version)
    wan_model.flash_attention = _sdpa
    attn.flash_attention = _sdpa
    for name in list(sys.modules):          # any other module that imported the symbol by name
        mod = sys.modules[name]
        if name.startswith("wan.") and getattr(mod, "flash_attention", None) is not None and mod is not attn:
            mod.flash_attention = _sdpa
    print("flash-attn missing; using PyTorch SDPA fallback", flush=True)


def build(model, local_attn=None, sink=None):
    from omegaconf import OmegaConf
    repo = REPO[model]
    sys.path.insert(0, repo)
    import wan.modules.causal_model  # noqa: F401  (load before patching)
    sdpa_fallback()
    if model == "sf":
        cfg = OmegaConf.merge(OmegaConf.load(f"{repo}/configs/default_config.yaml"),
                              OmegaConf.load(f"{repo}/configs/self_forcing_dmd.yaml"))
        cfg.model_kwargs.local_attn_size = 21 if local_attn is None else local_attn
        cfg.model_kwargs.sink_size = 0 if sink is None else sink
        from pipeline import CausalInferencePipeline
        pipe = CausalInferencePipeline(cfg, device="cuda")
        sd = torch.load(f"{CK}/self_forcing_dmd.pt", map_location="cpu")["generator_ema"]
        pipe.generator.load_state_dict(sd)
        run = lambda noise, prompts: pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
    elif model == "rf":
        from collections import OrderedDict
        cfg = OmegaConf.merge(OmegaConf.load(f"{repo}/configs/default_config.yaml"),
                              OmegaConf.load(f"{repo}/configs/rolling_forcing_dmd.yaml"))
        from pipeline import CausalInferencePipeline
        pipe = CausalInferencePipeline(cfg, device="cuda")
        sd = torch.load(f"{CK}/rolling_forcing_dmd.pt", map_location="cpu")["generator_ema"]
        sd = OrderedDict((k.replace("_fsdp_wrapped_module.", ""), v) for k, v in sd.items())
        pipe.generator.load_state_dict(sd)
        run = lambda noise, prompts: pipe.inference_rolling_forcing(noise=noise, text_prompts=prompts,
                                                                    return_latents=True, initial_latent=None)
    else:
        cfg = OmegaConf.load(f"{repo}/configs/longlive_inference_infinity.yaml")
        cfg.generator_ckpt = f"{CK}/longlive/models/longlive_base.pt"
        cfg.lora_ckpt = f"{CK}/longlive/models/lora.pt"
        cfg.distributed = False
        from pipeline import CausalInferencePipeline
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
        sd = torch.load(cfg.generator_ckpt, map_location="cpu")
        raw = sd["generator_ema" if cfg.use_ema else "generator"] if ("generator" in sd or "generator_ema" in sd) else sd["model"]
        pipe.generator.load_state_dict(raw)
        from utils.lora_utils import configure_lora_for_model
        import peft
        pipe.generator.model = configure_lora_for_model(pipe.generator.model, model_name="generator",
                                                        lora_config=cfg.adapter, is_main_process=True)
        lora = torch.load(cfg.lora_ckpt, map_location="cpu")
        peft.set_peft_model_state_dict(pipe.generator.model, lora["generator_lora"] if "generator_lora" in lora else lora)
        pipe.is_lora_enabled = True
        run = lambda noise, prompts: pipe.inference(noise=noise, text_prompts=prompts, return_latents=True,
                                                    low_memory=False, profile=False)
    pipe = pipe.to(dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda"); pipe.generator.to("cuda"); pipe.vae.to("cuda")
    return pipe, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sf", "rf", "ll"], required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--latents", type=int, default=720, help="latent frames; 720 -> 2877 px frames = 180 s at 16 fps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rundir", required=True, help="cwd holding wan_models/ symlink (repos load it relatively)")
    ap.add_argument("--local-attn", type=int, default=None, help="sf only: rolling window in latents (default 21)")
    ap.add_argument("--sink", type=int, default=None, help="sf only: sink size in latents (default 0)")
    ap.add_argument("--perturb-block", type=int, default=None, help="twin run: perturb the noise of this 3-latent block")
    ap.add_argument("--perturb-eps", type=float, nargs="+", default=[0.0],
                    help="twin perturbation sizes; 0 = exact rerun (numerical-noise control)")
    a = ap.parse_args()
    assert a.latents % 3 == 0
    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.rundir)
    torch.set_grad_enabled(False)
    import imageio.v2 as imageio            # fail fast before the model loads
    from einops import rearrange
    t0 = time.time()
    pipe, run = build(a.model, a.local_attn, a.sink)
    print(f"[{a.model}] pipeline ready {time.time() - t0:.0f}s", flush=True)
    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
    for pi, prompt in enumerate(prompts):
        for seed in a.seeds:
          for eps in (a.perturb_eps if a.perturb_block is not None else [None]):
            tag = "" if eps is None else f"_twin{a.perturb_block}e{eps:g}"
            stem = f"{a.model}_p{pi:02d}_s{seed}{tag}"
            mp4 = os.path.join(a.out, stem + ".mp4")
            if os.path.exists(mp4) and os.path.getsize(mp4) > 10_000:
                print("skip", stem, flush=True)
                continue
            torch.manual_seed(seed)
            noise = torch.randn([1, a.latents, 16, 60, 104], device="cuda", dtype=torch.bfloat16)
            if eps:
                # perturbation from a separate generator, so the global RNG stream (used inside the
                # pipeline for re-noising) is identical to the unperturbed run
                g = torch.Generator(device="cuda").manual_seed(10_000 + seed)
                b0 = 3 * a.perturb_block
                blk = noise[:, b0:b0 + 3].float()
                delta = torch.randn(blk.shape, device="cuda", generator=g)
                noise[:, b0:b0 + 3] = ((blk + eps * delta) / (1 + eps ** 2) ** 0.5).to(noise.dtype)
            t1 = time.time()
            video, latents = run(noise, [prompt])
            gen_s = time.time() - t1
            if hasattr(pipe.vae, "model") and hasattr(pipe.vae.model, "clear_cache"):
                pipe.vae.model.clear_cache()
            v = rearrange(video, "b t c h w -> b t h w c")[0]
            v = (255.0 * v.float().clamp(0, 1)).to(torch.uint8).cpu()
            w = imageio.get_writer(mp4, fps=16, codec="libx264", quality=8, macro_block_size=1)
            for fr in v.numpy():
                w.append_data(fr)
            w.close()
            torch.save(latents[0].to(torch.bfloat16).cpu()[::3].clone(), os.path.join(a.out, stem + ".lat3.pt"))
            json.dump({"model": a.model, "prompt_index": pi, "prompt": prompt, "seed": seed, "latents": a.latents,
                       "local_attn": a.local_attn, "sink": a.sink, "perturb_block": a.perturb_block, "perturb_eps": eps,
                       "frames": int(v.shape[0]), "gen_seconds": gen_s,
                       "noise_sha_head": float(noise.float().flatten()[:1000].sum())},
                      open(os.path.join(a.out, stem + ".json"), "w"))
            print(f"{stem} frames={v.shape[0]} gen={gen_s:.0f}s total={time.time() - t0:.0f}s", flush=True)
            del video, latents, v
            torch.cuda.empty_cache()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
