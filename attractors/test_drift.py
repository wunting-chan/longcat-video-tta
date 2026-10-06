"""CPU tests for drift.py. Run: python -m pytest -q test_drift.py"""
import torch

import drift as D

C, H, W = 4, 6, 8


def gen(seed):
    g = torch.Generator().manual_seed(seed)
    return lambda: torch.randn(1, 3, C, H, W, generator=g)


def opened(mode, alpha, k=2, seed=0, **kw):
    nb = gen(seed)
    att = torch.randn(C, H, W, generator=torch.Generator().manual_seed(99)) * 3
    corr = D.DriftCorrector(mode, alpha, att if mode != "anchor" else None, opening_blocks=k, **kw)
    blocks = [nb() for _ in range(k)]
    for i, b in enumerate(blocks):
        assert torch.equal(corr(b, 3 * i), b)            # opening blocks untouched
    return corr, nb, blocks


def frames(corr, coef_u, noise=0.3, seed=5):
    g = torch.Generator().manual_seed(seed)
    f = [corr.o + c * corr.u + noise * torch.randn(C, H, W, generator=g) for c in coef_u]
    return torch.stack(f)[None]


def proj(corr, f):
    return float(((f - corr.o) * corr.u).sum())


def test_opening_is_mean_of_first_k_blocks():
    corr, _, blocks = opened("repel", 1.0, k=3)
    assert torch.allclose(corr.o, torch.cat([b[0] for b in blocks]).mean(0), atol=1e-6)


def test_repel_removes_only_positive_projection():
    corr, _, _ = opened("repel", 1.0)
    x = frames(corr, [2.0, -2.0, 1.0])
    y = corr(x, 6)
    assert abs(proj(corr, y[0, 0])) < 1e-4 and abs(proj(corr, y[0, 2])) < 1e-4
    assert torch.allclose(y[0, 1], x[0, 1])
    orth = lambda f: (f - corr.o) - proj(corr, f) * corr.u
    assert torch.allclose(orth(y[0, 0]), orth(x[0, 0]), atol=1e-4)


def test_negative_alpha_pushes_toward_attractor():
    corr, _, _ = opened("repel", -0.5)
    x = frames(corr, [2.0, 2.0, 2.0])
    y = corr(x, 6)
    assert proj(corr, y[0, 0]) > proj(corr, x[0, 0]) + 0.9


def test_random_matches_norm_but_not_direction():
    rep, _, _ = opened("repel", 1.0)
    rnd, _, _ = opened("random", 1.0, rand_seed=7)
    x = frames(rep, [3.0, 1.0, 2.0])
    y_rep, y_rnd = rep(x.clone(), 6), rnd(x.clone(), 6)
    n_rep = (x - y_rep)[0].flatten(1).norm(dim=1)
    n_rnd = (x - y_rnd)[0].flatten(1).norm(dim=1)
    assert torch.allclose(n_rep, n_rnd, atol=1e-4)
    cos = torch.nn.functional.cosine_similarity((x - y_rnd)[0].flatten(1), rep.u.flatten()[None], dim=1)
    assert cos.abs().max() < 0.3                      # random direction, not the attractor axis


def test_random_is_reproducible_with_seed():
    a, _, _ = opened("random", 1.0, rand_seed=3)
    b, _, _ = opened("random", 1.0, rand_seed=3)
    x = frames(a, [2.0, 2.0, 2.0])
    assert torch.equal(a(x.clone(), 6), b(x.clone(), 6))


def test_anchor_restores_opening_stats():
    corr, nb, _ = opened("anchor", 1.0)
    y = corr(nb() * 2.5 + 1.7, 6)
    assert torch.allclose(y[0].mean((2, 3)), corr.mu0[None].expand(3, -1), atol=1e-4)
    assert torch.allclose(y[0].std((2, 3)), corr.sd0[None].expand(3, -1), atol=1e-3)


def test_both_applies_anchor_then_repel():
    corr, nb, _ = opened("both", 1.0)
    y = corr(frames(corr, [3.0, 3.0, 3.0]), 6)
    assert max(proj(corr, y[0, i]) for i in range(3)) < 1e-3


def test_window_limits_corrections():
    corr, _, _ = opened("repel", 1.0, win_lo=9, win_hi=12)
    x = frames(corr, [2.0, 2.0, 2.0])
    assert torch.equal(corr(x.clone(), 6), x)          # before window: untouched
    assert not torch.equal(corr(x.clone(), 9), x)      # inside window
    assert torch.equal(corr(x.clone(), 12), x)         # after window: untouched


def test_dtype_preserved():
    corr, _, _ = opened("repel", 1.0)
    x = frames(corr, [2.0, 2.0, 2.0]).to(torch.bfloat16)
    assert corr(x, 6).dtype == torch.bfloat16


def test_install_inserts_hook_before_step_3_2(tmp_path, monkeypatch):
    (tmp_path / "fakepipe.py").write_text("""
import torch
class P:
    def inference(self, n):
        output = torch.zeros(1, n * 3, 2)
        current_start_frame = 0
        for current_num_frames in [3] * n:
            denoised_pred = torch.ones(1, current_num_frames, 2)
            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            current_start_frame += current_num_frames
        return output
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    import fakepipe
    D.install(fakepipe.P)
    p = fakepipe.P()
    calls = []
    p._drift_hook = lambda x, s: (calls.append(s), x * 2)[1]
    assert torch.equal(p.inference(3), torch.full((1, 9, 2), 2.0))
    assert calls == [0, 3, 6]
    p._drift_hook = None
    assert torch.equal(p.inference(2), torch.ones(1, 6, 2))


def test_kick_modes_fixed_norm_and_direction():
    for mode, sign in (("kick_away", -1), ("kick_toward", 1)):
        corr, _, _ = opened(mode, 2.0, kick_unit=1.5, win_lo=6, win_hi=9)
        x = frames(corr, [0.5, 0.5, 0.5])
        y = corr(x.clone(), 6)
        d = (y - x)[0]
        assert torch.allclose(d.flatten(1).norm(dim=1), torch.full((3,), 3.0), atol=1e-4)
        assert abs(proj(corr, y[0, 0]) - proj(corr, x[0, 0]) - sign * 3.0) < 1e-3
        assert torch.equal(corr(x.clone(), 9), x)               # after the single chunk: no intervention
    corr, _, _ = opened("kick_random", 2.0, kick_unit=1.5, win_lo=6, win_hi=9, rand_seed=4)
    x = frames(corr, [0.5, 0.5, 0.5])
    d = (corr(x.clone(), 6) - x)[0]
    assert torch.allclose(d.flatten(1).norm(dim=1), torch.full((3,), 3.0), atol=1e-4)
    assert abs(float((d[0] * corr.u).sum())) < 1.5
