import numpy as np
import pytest
import lf
from lf import depth

ALPHAS = np.linspace(-1.5, 2.0, 15)


@pytest.fixture(scope="module")
def L():
    return lf.synthetic.synthetic_light_field(n_views=7, size=96)


def _interior(gt, d, margin=10, band=6):
    m = gt == d
    m &= ~depth.discontinuity_mask(gt, band)
    m[:margin] = m[-margin:] = False
    m[:, :margin] = m[:, -margin:] = False
    return m


@pytest.fixture(scope="module")
def results(L):
    return {"focus": depth.depth_from_focus(L, ALPHAS),
            "corr": depth.depth_from_correspondence(L, ALPHAS),
            "epi": depth.epi_structure_tensor_disparity(L),
            "comb": depth.depth_combined(L, ALPHAS)}


@pytest.mark.parametrize("name", ["focus", "corr", "epi", "comb"])
def test_layer_disparities(L, results, name):
    gt = L.meta["gt_disparity"]
    d = results[name]["disparity"]
    assert d.shape == gt.shape and d.dtype == np.float32
    for layer in (-1.0, 0.0, 1.5):
        m = _interior(gt, layer)
        assert m.sum() > 100
        assert np.abs(d[m] - layer).mean() < 0.2, (name, layer)


def test_output_keys_and_ranges(results):
    for k in ("focus", "corr"):
        r = results[k]
        assert r["cost_volume"].shape[0] == len(ALPHAS)
        assert 0 <= r["confidence"].min() and r["confidence"].max() <= 1


def test_evaluate_zero(L):
    gt = L.meta["gt_disparity"]
    e = depth.evaluate(gt, gt)
    assert e["mae"] == e["rmse"] == e["bad_pixel_ratio"] == 0
    assert e["mae_all"] == 0 and e["bad_pixel_ratio_all"] == 0


def test_evaluate_excludes_discontinuity_band(L):
    gt = L.meta["gt_disparity"]
    bad = gt + 1.0
    e = depth.evaluate(bad, gt)
    assert e["mae"] == pytest.approx(1.0) and e["bad_pixel_ratio"] == 1.0
    m = depth.discontinuity_mask(gt)
    assert m.any() and e["n_pixels"] == (~m).sum()


def test_disparity_to_depth():
    z = depth.disparity_to_depth(np.array([1.0, 2.0, 0.0]), baseline=2.0, focal_px=10.0)
    assert z[0] == pytest.approx(20.0) and z[1] == pytest.approx(10.0) and np.isinf(z[2])
    z0 = depth.disparity_to_depth(np.array([0.0]), 2.0, 10.0, d0_depth=5.0)
    assert z0[0] == pytest.approx(5.0)   # d=0 is the rectification plane


def test_epi_sign_convention(L):
    # nearer layer (+1.5) must come out positive, farther (-1) negative
    gt = L.meta["gt_disparity"]
    d = depth.epi_structure_tensor_disparity(L)["disparity"]
    assert d[_interior(gt, 1.5)].mean() > 1.0
    assert d[_interior(gt, -1.0)].mean() < -0.5
