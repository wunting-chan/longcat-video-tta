"""Drift correction for Self Forcing rollouts (attractor study, 2026-10-06).

A corrector is called on every freshly denoised latent block (B, F, C, H, W), between the
pipeline's Step 3.2 (store output) and Step 3.3 (rewrite the KV cache with the clean block), so
the correction changes both the saved video and what the model remembers.

Control acts in the generator's latent space; evaluation uses DINOv2 on decoded frames, so the
controller never optimises the metric it is judged by.

Opening o: mean latent of the first `opening_blocks` blocks (the prompt-specific reference);
no correction is applied to those blocks.

Modes (alpha may be negative for "toward the attractor")
  repel   d = c - o, u = d / |d|; for frame f, p = <f - o, u>; if p > 0: f <- f - alpha * p * u.
          Removes only the displacement toward the shared attractor c; other directions untouched.
          c = mean late latent of Self Forcing videos of OTHER prompts (estimated offline).
  anchor  per-channel mean/std pulled toward the opening's by fraction alpha.
  both    anchor, then repel (same alpha).
  random  matched control: the correction vector has exactly the norm repel would apply
          (|alpha * p+|, using the same u) but a random unit direction, drawn per frame from a
          seeded generator (independent of the generation RNG).
  kick_*  single-chunk perturbation of fixed norm m = alpha * kick_unit added to every frame of the
          window blocks: away = -m u, toward = +m u, random = m r (r a seeded random unit vector).
          kick_unit (frozen from prompts 0-7) = mean displacement toward c accumulated by 30 s.
Window: corrections are applied only to blocks whose start latent index lies in [win_lo, win_hi).
"""
from __future__ import annotations

import inspect
import textwrap

import torch

HOOK_LINE = "            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred\n"
MODES = ("repel", "anchor", "both", "random", "kick_away", "kick_toward", "kick_random")


class DriftCorrector:
    def __init__(self, mode: str, alpha: float, attractor: torch.Tensor | None = None,
                 opening_blocks: int = 3, win_lo: int = 0, win_hi: int = 10**9, rand_seed: int = 0,
                 kick_unit: float = 0.0):
        assert mode in MODES
        if mode != "anchor":
            assert attractor is not None and attractor.dim() == 3, "attractor must be (C, H, W)"
        self.mode, self.alpha = mode, float(alpha)
        self.c = None if attractor is None else attractor.float()
        self.k, self.win = int(opening_blocks), (int(win_lo), int(win_hi))
        self.rand_seed = rand_seed
        self.kick_unit = float(kick_unit)
        if mode.startswith("kick"):
            assert self.kick_unit > 0, "kick modes need kick_unit"
        self.reset()

    def reset(self, rand_seed: int | None = None):
        self.buf, self.o, self.u = [], None, None
        self.log = []   # per block: (start, p_before, p_after, correction_norm)
        self.g = None
        if rand_seed is not None:
            self.rand_seed = rand_seed

    def _finish_opening(self):
        X = torch.cat(self.buf, 0)                       # (k*F, C, H, W)
        self.o = X.mean(0)
        self.mu0 = X.mean((2, 3)).mean(0)
        self.sd0 = X.std((2, 3)).mean(0)
        if self.c is not None:
            d = (self.c.to(self.o.device) - self.o).flatten()
            self.u = (d / d.norm()).view_as(self.o)

    def _anchor(self, x):
        mu = x.mean((2, 3)); sd = x.std((2, 3)).clamp(min=1e-6)
        sd_new = sd + self.alpha * (self.sd0[None] - sd)
        mu_new = mu + self.alpha * (self.mu0[None] - mu)
        return (x - mu[..., None, None]) / sd[..., None, None] * sd_new[..., None, None] + mu_new[..., None, None]

    def _proj(self, x):
        return ((x - self.o[None]) * self.u[None]).flatten(1).sum(1)

    def __call__(self, pred: torch.Tensor, start: int) -> torch.Tensor:
        x = pred.float()[0]                              # (F, C, H, W)
        if self.o is None:
            self.buf.append(x)
            if len(self.buf) >= self.k:
                self._finish_opening()
            self.log.append((start, 0.0, 0.0, 0.0))
            return pred
        if not (self.win[0] <= start < self.win[1]):
            p = self._proj(x) if self.u is not None else torch.zeros(len(x))
            self.log.append((start, float(p.mean()), float(p.mean()), 0.0))
            return pred
        y, cn = x, torch.zeros(len(x), device=x.device)
        if self.mode.startswith("kick"):
            m = self.alpha * self.kick_unit
            if self.mode == "kick_random":
                if self.g is None:
                    self.g = torch.Generator(device="cpu").manual_seed(self.rand_seed)
                v = torch.randn(self.o.shape, generator=self.g).to(y.device)
                v = v / v.norm()
            else:
                v = self.u if self.mode == "kick_toward" else -self.u
            y = y + m * v[None]
            cn = torch.full((len(x),), m, device=x.device)
        if self.mode in ("anchor", "both"):
            y = self._anchor(y)
        if self.mode in ("repel", "both", "random"):
            p = self._proj(y)
            mag = self.alpha * p.clamp(min=0)                            # (F,)
            if self.mode == "random":
                if self.g is None:
                    self.g = torch.Generator(device="cpu").manual_seed(self.rand_seed)
                r = torch.randn(y.shape, generator=self.g).to(y.device)
                r = r / r.flatten(1).norm(dim=1)[:, None, None, None]
                y = y - mag[:, None, None, None] * r
            else:
                y = y - mag[:, None, None, None] * self.u[None]
            cn = mag.abs()
        p0 = self._proj(x) if self.u is not None else torch.zeros(len(x))
        p1 = self._proj(y) if self.u is not None else torch.zeros(len(x))
        self.log.append((start, float(p0.mean()), float(p1.mean()), float(cn.mean())))
        return y[None].to(pred.dtype)


def install(pipeline_cls) -> None:
    """Re-define pipeline_cls.inference with one hook call before Step 3.2. The third-party file is
    not modified; the method is recompiled from its own source in its own module namespace."""
    src = textwrap.dedent(inspect.getsource(pipeline_cls.inference))
    line = textwrap.dedent(HOOK_LINE).rstrip("\n")
    body_line = [l for l in src.splitlines() if l.strip() == line.strip()]
    assert len(body_line) == 1, "hook anchor not found exactly once; pipeline changed?"
    indent = body_line[0][: len(body_line[0]) - len(body_line[0].lstrip())]
    hook = (f"{indent}if getattr(self, '_drift_hook', None) is not None:\n"
            f"{indent}    denoised_pred = self._drift_hook(denoised_pred, current_start_frame)\n")
    src = src.replace(body_line[0] + "\n", hook + body_line[0] + "\n", 1)
    mod = inspect.getmodule(pipeline_cls)
    ns = dict(vars(mod))
    exec(compile(src, f"<drift-patched {mod.__name__}.inference>", "exec"), ns)
    pipeline_cls.inference = ns["inference"]
