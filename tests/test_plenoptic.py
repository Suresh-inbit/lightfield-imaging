import numpy as np
import pytest
from lf.synthetic import synthetic_light_field
from lf import plenoptic as P, refocus as R

LF = synthetic_light_field(n_views=7, size=96)


@pytest.mark.parametrize("invert", [False, True])
def test_roundtrip_exact(invert):
    raw = P.to_lenslet(LF, invert=invert)
    assert raw.shape == (7 * 96, 7 * 96, 3)
    back = P.from_lenslet(raw, (7, 7), invert=invert)
    assert np.array_equal(back.views, LF.views)


def test_layout_and_invert():
    raw = P.to_lenslet(LF)
    assert np.array_equal(raw[2 * 7 + 3, 5 * 7 + 1], LF.views[3, 1, 2, 5])   # pixel (i,j)=(3,1) of lens (2,5)
    raw_i = P.to_lenslet(LF, invert=True)
    assert np.array_equal(raw_i[2 * 7 + 3, 5 * 7 + 1], LF.views[6 - 3, 6 - 1, 2, 5])


def test_closeup_shape():
    raw = P.to_lenslet(LF)
    assert P.lenslet_closeup(raw, (7, 7), (48, 48), 6).shape == (42, 42, 3)
    assert P.lenslet_closeup(raw, (7, 7), (0, 0), 6).shape == (42, 42, 3)


def test_conventional_equals_refocus_alpha0():
    assert np.allclose(P.conventional_photo(LF), R.refocus(LF, 0.0), atol=1e-5)
    raw = P.to_lenslet(LF)
    assert np.allclose(P.conventional_photo(raw, (7, 7)), R.refocus(LF, 0.0), atol=1e-5)


@pytest.mark.parametrize("radius", [0, 1.0, 2.0])
def test_stopdown_equals_masked_refocus(radius):
    m = P.disc_mask(7, 7, radius).astype(np.float32)
    assert np.allclose(P.conventional_photo(LF, radius=radius), R.refocus(LF, 0.0, mask=m), atol=1e-5)
    if radius == 0:
        assert np.allclose(P.conventional_photo(LF, radius=0), LF.center_view, atol=1e-6)


def test_design_ng_prototype():
    d = P.design(n_microlenses=(292, 292))
    assert d["angular_resolution"] == (14, 14)
    assert d["n_microlenses"] == (292, 292)
    d = P.design()
    assert d["angular_resolution"] == (14, 14)
    assert d["microlens_f_number"] == pytest.approx(4.0)
    assert d["f_numbers_match"] and d["overlap_fraction"] == 0.0
    assert d["microlens_image_diameter_um"] == pytest.approx(125.0)


def test_design_fnumber_mismatch():
    assert P.design(main_f_number=2.8)["overlap_fraction"] > 0.1       # cross-talk
    assert not P.design(main_f_number=2.8)["f_numbers_match"]
    small = P.design(main_f_number=8.0)
    assert small["overlap_fraction"] == 0 and small["wasted_fraction"] > P.design()["wasted_fraction"]


def test_resolution_tradeoff_and_table():
    assert P.resolution_tradeoff((2656, 1216), (16, 16)) == (166, 76)
    assert P.resolution_tradeoff((4044, 3240), (5, 4)) == (808, 810)
    assert P.resolution_tradeoff((1629, 2052), (9, 9)) == (181, 228)
    for row in P.camera_table()[1:]:
        assert row["spatial_computed"] == row["spatial_slide"]


def test_fnumber_mismatch_sim():
    ref = P.to_lenslet(LF)
    matched = P.simulate_fnumber_mismatch(LF, 4.0)
    small = P.simulate_fnumber_mismatch(LF, 8.0)
    big = P.simulate_fnumber_mismatch(LF, 2.8)
    assert matched.shape == small.shape == big.shape == ref.shape
    assert (small.sum(-1) == 0).mean() > (matched.sum(-1) == 0).mean()
