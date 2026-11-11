import numpy as np
import pytest
from lf import hdr as H

GT = H.synthetic_hdr_light_field(n_views=5, size=64, seed=0)
EXPS = (1 / 32, 1.0, 32.0)


def test_synthetic_hdr_lf():
    assert GT.meta["radiance"] is True
    assert GT.meta["dynamic_range"] > 500
    assert GT.meta["gt_disparity"].shape == (64, 64)
    assert GT.views.dtype == np.float32 and GT.views.max() > 1.0


def test_weight_hat():
    w = H.weight_hat(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(w, [0, 0.5, 1, 0.5, 0])


def test_solve_response_recovers_gamma():
    rng = np.random.default_rng(0)
    ex = [1 / 32, 1 / 8, 1 / 2, 2, 8]
    c = H.synthetic_hdr_light_field(3, 96).center_view
    imgs = [H.simulate_capture(c, t, rng=rng) for t in ex]
    g, lnE = H.solve_response(H.sample_pixels(imgs, 200, rng), ex)
    z = np.arange(256)
    sel = (z > 8) & (z < 250)
    ref = 2.2 * np.log(np.maximum(z, 1) / 255.0)
    assert np.corrcoef(g[sel], ref[sel])[0, 1] > 0.99


def test_merge_hdr_linear_noise_free():
    rad = GT.center_view
    exps = [1 / 64, 1 / 8, 1.0, 8.0, 64.0]
    imgs = [H.simulate_capture(rad, t, crf="linear", photon_scale=0, read_noise=0, bits=16) for t in exps]
    m = H.merge_hdr(imgs, exps)
    ok = (rad > 0.02) & (rad < 10)
    assert np.all(np.abs(m[ok] / rad[ok] - 1) < 0.02)
    assert np.isfinite(m).all()


def test_merge_all_saturated_is_finite():
    img = np.ones((4, 4, 3), np.float32)
    m = H.merge_hdr([img, img], [1.0, 2.0])
    assert np.isfinite(m).all() and np.allclose(m, 1.0, atol=0.03)       # shortest exposure bound


@pytest.mark.parametrize("method", ["reinhard", "log", "gamma"])
def test_tonemap_range(method):
    t = H.tonemap(GT.center_view, method=method)
    assert t.shape == GT.center_view.shape and t.min() >= 0 and t.max() <= 1


def test_exposure_fusion():
    imgs = [H.simulate_capture(GT.center_view, t, photon_scale=0, read_noise=0) for t in EXPS]
    f = H.exposure_fusion(imgs)
    assert f.shape == imgs[0].shape and f.min() >= 0 and f.max() <= 1


def test_exposure_pattern_contains_all():
    for kind in ["checker", "rows", "radial", "random"]:
        m = H.exposure_pattern(5, 5, kind, EXPS)
        assert m.shape == (5, 5) and set(np.unique(m)) == set(EXPS)


def test_strategy_a_beats_single_exposures_and_near_c():
    a = H.strategy_a(GT, "checker")
    c = H.temporal_bracket_hdr_lf(GT, EXPS)
    singles = []
    for t in EXPS:
        l, _ = H.capture_bracketed_views(GT, np.full((5, 5), t))
        singles.append(H.log_rmse(l.views ** 2.2 / t, GT.views))
    assert a["log_rmse"] < 0.6 * min(singles)
    assert c["log_rmse"] < a["log_rmse"] < 4 * c["log_rmse"]
    assert a["shots"] == 1 and c["shots"] == 3
    nod = H.strategy_a(GT, "checker", disparity=None)
    assert nod["log_rmse"] > a["log_rmse"]


def test_hdr_refocus_beats_ldr_refocus():
    c = H.temporal_bracket_hdr_lf(GT, EXPS)
    for alpha in (0.0, 1.5):
        r = H.ldr_refocus_comparison(GT, alpha, 1.0, hdr_lf=c["hdr_lf"])
        assert r["log_rmse_hdr"] < r["log_rmse_ldr"]
        assert r["log_rmse_hdr"] < r["log_rmse_ldr_lin"]
    img = H.hdr_refocus(GT, 0.0)
    assert img.min() >= 0 and img.max() <= 1


def test_aperture_area_bracketing_runs():
    d = H.aperture_area_bracketing(GT, (1, 3, 9), alpha=0.0)
    assert d["image"].shape == (64, 64, 3) and d["shots"] == 3 and np.isfinite(d["log_rmse"])


def test_pseudo_hdr():
    from lf.synthetic import synthetic_light_field
    p = H.hdr_from_ldr_light_field(synthetic_light_field(3, 48))
    assert p.meta["radiance"] and p.views.max() > 1
