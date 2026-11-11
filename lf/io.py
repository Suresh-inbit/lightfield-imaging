"""Loaders for the light-field datasets used in the project.

All loaders return a :class:`lf.LightField` with ``views`` of shape
``(U, V, H, W, C)`` in float32 [0, 1].  See ``scripts/download_data.py`` for
where the data comes from.
"""
from __future__ import annotations
import re, pathlib
import numpy as np
import imageio.v3 as iio
from .lightfield import LightField

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Stanford (new) archive rectified files look like
#   out_00_00_-1206.126709_-1206.126709.png   ->  row, col, cam_y_mm, cam_x_mm
_STANFORD_RE = re.compile(r"out_(\d+)_(\d+)_(-?[\d.]+)_(-?[\d.]+)_?\.png$")  # some sets have a trailing "_"


def _to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def load_stanford_rectified(folder: str | pathlib.Path, downsample: int = 1,
                            angular_crop: int | None = None, name: str | None = None) -> LightField:
    """Load a Stanford Light Field Archive ``rectified`` folder (17 x 17 views).

    The archive stores one PNG per camera position, named with the grid index
    and the physical camera coordinates (mm).  The gantry moves on a plane, so
    this is exactly the two-plane ``(u, v) x (s, t)`` sampling of Ng 2005 with a
    *large* baseline (the "Stanford camera array" / gantry on slide 16).

    Parameters
    ----------
    folder : path containing the ``out_*.png`` files (or its parent).
    downsample : integer spatial downsampling factor applied while loading
        (1024 x 1024 x 289 views is ~3.6 GB as float32; use 4 for interactive work).
    angular_crop : keep only the central ``n x n`` views.
    """
    folder = pathlib.Path(folder)
    files = sorted(folder.rglob("out_*.png"))
    if not files:
        raise FileNotFoundError(f"no out_*.png under {folder}; run scripts/download_data.py")
    entries = []
    for f in files:
        m = _STANFORD_RE.search(f.name)
        if m:
            r, c, y, x = int(m[1]), int(m[2]), float(m[3]), float(m[4])
            entries.append((r, c, y, x, f))
    rows = sorted({e[0] for e in entries}); cols = sorted({e[1] for e in entries})
    U, V = len(rows), len(cols)
    if angular_crop is not None:
        ku = [rows[U // 2 - angular_crop // 2 + i] for i in range(angular_crop)]
        kv = [cols[V // 2 - angular_crop // 2 + i] for i in range(angular_crop)]
        entries = [e for e in entries if e[0] in ku and e[1] in kv]
        rows, cols = ku, kv
        U, V = angular_crop, angular_crop
    rmap = {r: i for i, r in enumerate(rows)}; cmap = {c: i for i, c in enumerate(cols)}
    views = None
    cam_y = np.zeros((U, V)); cam_x = np.zeros((U, V))
    for r, c, y, x, f in entries:
        img = _to_float(iio.imread(f))
        if img.ndim == 2:
            img = img[..., None]
        img = img[..., :3]
        if downsample > 1:
            H, W = (img.shape[0] // downsample) * downsample, (img.shape[1] // downsample) * downsample
            img = img[:H, :W].reshape(H // downsample, downsample, W // downsample, downsample, -1).mean(axis=(1, 3))
        if views is None:
            views = np.zeros((U, V) + img.shape, dtype=np.float32)
        views[rmap[r], cmap[c]] = img
        cam_y[rmap[r], cmap[c]] = y; cam_x[rmap[r], cmap[c]] = x
    # Orientation.  Our convention (docs/conventions.md) is that *nearer* points
    # have *positive* disparity, i.e. a point at disparity d sits at
    # t + d (v - v0).  A camera moving in +x makes near objects move towards -t,
    # so ``v`` must run opposite to the gantry's x axis, and likewise ``u``
    # opposite to y.  Verified empirically on Lego Knights (knights sharp at
    # alpha > 0, castle wall at alpha < 0; horizontal-only and vertical-only
    # shift-and-add peak at the same alpha).  See docs/01_datasets.md.
    if V > 1 and cam_x[0, -1] > cam_x[0, 0]:
        views = views[:, ::-1]; cam_x = cam_x[:, ::-1]; cam_y = cam_y[:, ::-1]
    if U > 1 and cam_y[-1, 0] > cam_y[0, 0]:
        views = views[::-1]; cam_x = cam_x[::-1]; cam_y = cam_y[::-1]
    meta = {"source": "Stanford Light Field Archive (lightfield.stanford.edu)",
            "cam_x_mm": cam_x, "cam_y_mm": cam_y, "downsample": downsample}
    return LightField(np.ascontiguousarray(views), name or folder.parent.name if folder.name == "rectified" else folder.name, meta)


def load_lytro_eslf(path: str | pathlib.Path, angular: tuple[int, int] = (14, 14),
                    crop_angular: int | None = 9, downsample: int = 1, name: str | None = None,
                    gain: float | str | None = "auto") -> LightField:
    """Load an ESLF PNG from the Stanford Lytro Illum archive.

    An ESLF ("extracted sub-aperture light field") image is the *lenslet*
    representation: the ``14 x 14`` pixels under each micro-lens are stored as
    a ``14 x 14`` block, blocks tiled in micro-lens order.  Pixel ``(i, j)`` of
    the PNG is therefore ``L[u = i % 14, v = j % 14, s = i // 14, t = j // 14]``.
    This is exactly what the raw sensor of a plenoptic camera records (after
    hexagonal-to-square resampling and demosaicing), so decoding it *is*
    "extracting sub-aperture images" (Ng 2005, slide 41).

    The outer views of a Lytro Illum are strongly vignetted (dark / noisy), so by
    default we keep the central ``9 x 9``.  The archive PNGs are also dark
    (mean intensity ~0.1); ``gain='auto'`` rescales so the 99.5th percentile of
    the central view maps to 1 (clipping the few brightest pixels), ``gain=None``
    keeps the raw values, a float applies that factor.  The gain used is stored
    in ``meta['gain']``.
    """
    path = pathlib.Path(path)
    img = _to_float(iio.imread(path))
    if img.ndim == 2:
        img = img[..., None]
    img = img[..., :3]
    au, av = angular
    H, W = (img.shape[0] // au) * au, (img.shape[1] // av) * av
    img = img[:H, :W]
    lf = img.reshape(H // au, au, W // av, av, -1).transpose(1, 3, 0, 2, 4)  # (u, v, s, t, c)
    out = LightField(np.ascontiguousarray(lf), name or path.stem,
                     {"source": "Stanford Lytro Light Field Archive (lightfields.stanford.edu)",
                      "lenslet_png": str(path)})
    if crop_angular:
        out = out.crop_angular(crop_angular)
    if gain is not None:
        g = float(1.0 / max(np.percentile(out.center_view, 99.5), 1e-6)) if gain == "auto" else float(gain)
        out = LightField(np.clip(out.views * g, 0, 1), out.name, {**out.meta, "gain": g})
    if downsample > 1:
        out = out.downsample(downsample)
    return out


def load(name: str, **kw) -> LightField:
    """Convenience: ``load('lego')``, ``load('bracelet')``, ``load('lytro')``,
    ``load('synthetic')``.  Keyword args go to the specific loader."""
    key = name.lower()
    if "synth" in key:
        from .synthetic import synthetic_light_field
        return synthetic_light_field(**kw)
    if "lego" in key:
        return load_stanford_rectified(RAW / "stanford_lego_knights", name="lego_knights", **kw)
    if "bracelet" in key:
        return load_stanford_rectified(RAW / "stanford_bracelet", name="bracelet", **kw)
    if "lytro" in key or "flower" in key:
        return load_lytro_eslf(RAW / "lytro_illum_flowers" / "flowers_plants_1_eslf.png", **kw)
    raise KeyError(name)


def available() -> dict[str, bool]:
    """Which datasets are present on disk."""
    return {
        "synthetic": True,
        "lego": (RAW / "stanford_lego_knights" / ".done").exists(),
        "bracelet": (RAW / "stanford_bracelet" / ".done").exists(),
        "lytro": (RAW / "lytro_illum_flowers" / ".done").exists(),
    }
