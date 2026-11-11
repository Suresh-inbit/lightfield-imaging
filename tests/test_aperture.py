import numpy as np
from lf.synthetic import synthetic_light_field
from lf import aperture as A
from lf.refocus import refocus

LF = synthetic_light_field(n_views=7, size=96)


def test_mask_shapes():
    m = A.aperture_mask(7, 7, "circle", 1.0)
    assert m.shape == (7, 7) and m.sum() == 5 and m[3, 3] == 1
    assert A.aperture_mask(7, 7, "square", 1.0).sum() == 9
    an = A.aperture_mask(7, 7, "annulus", 3.0)
    assert an[3, 3] == 0 and an.sum() > 0
    s = A.aperture_mask(7, 7, "circle", 2.0, soft=True)
    assert s.min() >= 0 and s.max() <= 1


def test_stop_down_center_view():
    for a in (0.0, 1.5):
        out = A.stop_down(LF, 0.0, alpha=a)
        assert np.allclose(out, LF.center_view, atol=1e-6)


def test_stop_down_full_equals_refocus():
    out = A.stop_down(LF, 100.0, alpha=0.5)
    assert np.allclose(out, refocus(LF, 0.5), atol=1e-6)


def test_synthesize_integer_and_fraction():
    for u, v in [(0, 0), (3, 3), (6, 2)]:
        assert np.array_equal(A.synthesize_view(LF, u, v), LF.view(u, v))
    mid = A.synthesize_view(LF, 2.5, 3)
    assert np.allclose(mid, 0.5 * (LF.view(2, 3) + LF.view(3, 3)), atol=1e-6)


def test_effective_f_number():
    assert A.effective_f_number(4, 14, 0) == 4 * 14
    assert A.effective_f_number(4, 14, 100) == 4
    assert A.effective_f_number(4, 14, 3) == 4 * 14 / 7


def test_sequence_and_dolly():
    for p in ("circle", "row", "column", "diagonal"):
        seq = A.perspective_sequence(LF, p, n=8)
        assert len(seq) == 8 and seq[0].shape == (96, 96, 3)
    assert A.dolly(LF, 0.0, 1.3).shape == (96, 96, 3)
