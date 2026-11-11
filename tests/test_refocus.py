import numpy as np
import pytest
from lf.synthetic import synthetic_light_field
from lf import refocus as R

LF = synthetic_light_field(n_views=7, size=96)
# ROIs inside each layer (rows, cols); text rect 10-58, gravel disc centre (65,63) r=19
ROIS = {0.0: (slice(15, 45), slice(15, 40)), 1.5: (slice(60, 72), slice(58, 70)),
        -1.0: (slice(66, 92), slice(2, 36))}
ALPHAS = np.arange(-2.0, 2.51, 0.25)


@pytest.mark.parametrize("method", ["subpixel", "integer", "fourier"])
def test_layers_sharpest_at_own_alpha(method):
    for a_true, roi in ROIS.items():
        s = R.sweep_sharpness(LF, ALPHAS, roi=roi, method=method)
        assert abs(ALPHAS[np.argmax(s)] - a_true) <= 0.25, (method, a_true, s)


def test_mask_none_equals_ones():
    for m in ["subpixel", "integer", "fourier"]:
        a = R.refocus(LF, 0.7, method=m)
        b = R.refocus(LF, 0.7, method=m, mask=np.ones((7, 7)))
        assert np.allclose(a, b, atol=1e-6)


def test_methods_agree_roughly_and_shape():
    a = R.refocus(LF, 1.0, method="subpixel")
    b = R.refocus(LF, 1.0, method="fourier")
    assert a.shape == (96, 96, 3)
    assert np.allclose(a[10:-10, 10:-10], b[10:-10, 10:-10], atol=1e-3)  # integer shifts


def test_alpha_zero_is_mean():
    assert np.allclose(R.refocus(LF, 0.0), LF.views.mean(axis=(0, 1)), atol=1e-6)


def test_focal_stack_and_shear():
    fs = R.focal_stack(LF, [0, 1])
    assert fs.shape == (2, 96, 96, 3)
    assert np.allclose(R.shear(LF, 1.0).mean(axis=(0, 1)), R.refocus(LF, 1.0), atol=1e-4)


def test_ng_alpha_roundtrip():
    a = np.array([0.5, 0.8, 1.0, 1.3, 2.0])
    for p in (1.0, 3.5):
        d = R.ng_alpha_to_disparity(a, p)
        assert np.allclose(R.disparity_to_ng_alpha(d, p), a)
    assert R.ng_alpha_to_disparity(1.0) == 0.0
