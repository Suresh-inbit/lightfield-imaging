"""Upload loaders: array, lenslet image and ZIP of views."""
import io, zipfile
import numpy as np
import imageio.v3 as iio
import pytest
import lf
from lf.plenoptic import to_lenslet


@pytest.fixture(scope="module")
def L():
    return lf.synthetic.synthetic_light_field(n_views=5, size=48)


def _png(img):
    return iio.imwrite("<bytes>", lf.utils.to_uint8(img), extension=".png")


def test_npy_roundtrip(L):
    buf = io.BytesIO(); np.save(buf, L.views)
    out = lf.io.load_upload(buf.getvalue(), "mine.npy")
    assert out.shape == L.shape and np.allclose(out.views, L.views, atol=1e-6)
    out = lf.io.load_upload(buf.getvalue(), "mine.npy", crop_angular=3, downsample=2)
    assert out.shape == (3, 3, 24, 24, 3)


def test_lenslet_image(L):
    raw = to_lenslet(L)
    out = lf.io.load_upload(_png(raw), "raw.png", angular=(5, 5), crop_angular=None, gain=None)
    assert out.shape == L.shape and np.abs(out.views - L.views).max() < 1 / 255 + 1e-6
    out = lf.io.load_upload(_png(raw), "raw.png", angular=(5, 5), crop_angular=3, gain=None)
    assert np.abs(out.views - L.crop_angular(3).views).max() < 1 / 255 + 1e-6


@pytest.mark.parametrize("pattern", ["view_{u:02d}_{v:02d}.png", "img{i:03d}.png",
                                     "out_{u:02d}_{v:02d}_{y:.6f}_{x:.6f}_.png"])
def test_zip_of_views(L, pattern):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for u in range(L.nu):
            for v in range(L.nv):
                # Stanford convention: coordinates decrease with our angular index
                z.writestr("sub/" + pattern.format(u=u, v=v, i=u * L.nv + v, y=-10.0 * u, x=-10.0 * v),
                           _png(L.view(u, v)))
    out = lf.io.load_upload(buf.getvalue(), "views.zip")
    assert out.shape == L.shape and np.abs(out.views - L.views).max() < 1 / 255 + 1e-6


def test_rejects_bad_input(L):
    with pytest.raises(ValueError):
        lf.io.load_upload(b"x", "a.txt")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(6):
            z.writestr(f"a{chr(97 + i)}.png", _png(L.view(0, 0)))
    with pytest.raises(ValueError):
        lf.io.load_upload(buf.getvalue(), "views.zip")
    with pytest.raises(MemoryError):
        lf.io.from_lenslet_image(np.zeros((400, 400, 3), np.uint8), angular=(2, 2), crop_angular=None, max_mb=0.1)
