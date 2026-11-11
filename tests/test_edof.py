import numpy as np
import pytest
import lf
from lf import edof, refocus
from lf.utils import gray

ALPHAS = np.linspace(-1.5, 2.0, 15)


def lap_var(img):
    import cv2
    return float(cv2.Laplacian(gray(img).astype(np.float32), cv2.CV_32F).var())


@pytest.fixture(scope="module")
def L():
    return lf.synthetic.synthetic_light_field(n_views=7, size=96)


@pytest.fixture(scope="module")
def res(L):
    return edof.all_in_focus(L, ALPHAS)


def test_shapes_and_index_map(L, res):
    assert res["image"].shape == (96, 96, 3)
    im = res["index_map"]
    assert im.shape == (96, 96) and np.issubdtype(im.dtype, np.integer)
    assert im.min() >= 0 and im.max() < len(ALPHAS)
    assert res["focus_measure"].shape == (len(ALPHAS), 96, 96)
    assert np.allclose(res["disparity"], ALPHAS[im])


def test_sharper_than_every_refocus(L, res):
    s = lap_var(res["image"])
    stack = refocus.focal_stack(L, ALPHAS)
    assert s > max(lap_var(x) for x in stack)


def test_matches_center_in_text_region(L, res):
    gt = L.meta["gt_disparity"]
    m = (gt == 0.0)
    m[:12] = m[-12:] = False; m[:, :12] = m[:, -12:] = False
    from scipy.ndimage import binary_erosion
    m = binary_erosion(m, iterations=6)
    assert m.sum() > 100
    err = np.abs(res["image"] - L.center_view)[m].mean()
    assert err < 0.03


@pytest.mark.parametrize("sm", ["gaussian", "none"])
def test_smoothing_modes(L, sm):
    r = edof.all_in_focus(L, ALPHAS, smoothing=sm)
    assert r["image"].shape == (96, 96, 3) and r["index_map"].max() < len(ALPHAS)


def test_compare_with_stop_down(L):
    full, stopped, aif = edof.compare_with_stop_down(L, ALPHAS, small_radius=1, noise_sigma=0.02)
    assert full.shape == stopped.shape == aif.shape == (96, 96, 3)
    clean = edof.compare_with_stop_down(L, ALPHAS, small_radius=1)[1]
    assert np.abs(stopped - clean).mean() > 0.005          # noise was added
