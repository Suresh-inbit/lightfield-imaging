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
    # crop the angular grid on the integer data before converting to float:
    # a full ESLF is 5264 x 7574 px, i.e. ~480 MB as float32 if converted first
    import cv2                                   # decodes straight to 3 channels: ~half the peak memory of PIL
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    raw = iio.imread(path) if raw is None else raw[..., ::-1]              # BGR -> RGB as a view
    out = from_lenslet_image(raw, angular=angular, crop_angular=crop_angular,
                             downsample=downsample, gain=gain, name=name or path.stem, max_mb=2000.0)
    out.meta.update({"source": "Stanford Lytro Light Field Archive (lightfields.stanford.edu)",
                     "lenslet_png": str(path)})
    return out


# light fields shipped in the repository: key -> (file, (U, V)); see data/bundled/README.md
BUNDLED = {"lytro_people": (ROOT / "data" / "bundled" / "lytro_people_9x9.png", (9, 9))}


def load_view_mosaic(path: str | pathlib.Path, angular: tuple[int, int], crop_angular: int | None = None,
                     downsample: int = 1, gain: float | str | None = "auto", name: str | None = None,
                     **_ignored) -> LightField:
    """Load a light field stored as one image of ``U x V`` tiled sub-aperture
    views (row-major).  Same ``gain`` semantics as :func:`load_lytro_eslf`."""
    path = pathlib.Path(path)
    img = _to_float(iio.imread(path))[..., :3]
    U, V = angular
    H, W = img.shape[0] // U, img.shape[1] // V
    views = img[:U * H, :V * W].reshape(U, H, V, W, -1).transpose(0, 2, 1, 3, 4)
    out = LightField(np.ascontiguousarray(views), name or path.stem,
                     {"source": "bundled sample (Stanford Lytro Light Field Archive)", "file": str(path)})
    if crop_angular and crop_angular < min(U, V):
        out = out.crop_angular(crop_angular)
    if gain is not None:
        g = float(1.0 / max(np.percentile(out.center_view, 99.5), 1e-6)) if gain == "auto" else float(gain)
        np.multiply(out.views, g, out=out.views); np.clip(out.views, 0, 1, out=out.views)
        out.meta["gain"] = g
    return out.downsample(downsample) if downsample > 1 else out


def load(name: str, **kw) -> LightField:
    """Convenience: ``load('lego')``, ``load('bracelet')``, ``load('lytro')``,
    ``load('synthetic')``.  Keyword args go to the specific loader."""
    key = name.lower()
    if "synth" in key:
        from .synthetic import synthetic_light_field
        return synthetic_light_field(**kw)
    if "lego" in key:
        return load_stanford_rectified(RAW / "stanford_lego_knights", name="lego_knights", **kw)
    if "lytro_illum_flowers" in key or key in ("lytro", "flowers"):
        return load_lytro_eslf(RAW / "lytro_illum_flowers" / "flowers_plants_1_eslf.png", **kw)
    if key in BUNDLED and not (RAW / key / ".done").exists():   # shipped with the repository
        return load_view_mosaic(BUNDLED[key][0], BUNDLED[key][1], name=key, **kw)
    for d in (RAW / f"stanford_{key}", RAW / key):          # any other downloaded Stanford scene
        if (d / ".done").exists() and d.name.startswith("stanford_"):
            return load_stanford_rectified(d, name=d.name.removeprefix("stanford_"), **kw)
    for d in (RAW / key, RAW / f"lytro_{key}"):             # any other downloaded Lytro sample
        if (d / ".done").exists():
            pngs = sorted(d.glob("*_eslf.png"))
            if pngs:
                return load_lytro_eslf(pngs[0], **kw)
    raise KeyError(f"{name!r}: not downloaded (see docs/dataset_links.md and scripts/download_data.py --list)")


def available() -> dict[str, bool]:
    """Which datasets are present on disk.  The three default sets are always
    listed; other downloaded sets appear under their short key."""
    out = {
        "synthetic": True,
        "lego": (RAW / "stanford_lego_knights" / ".done").exists(),
        "bracelet": (RAW / "stanford_bracelet" / ".done").exists(),
        "lytro": (RAW / "lytro_illum_flowers" / ".done").exists(),
    }
    for k, (f, _) in BUNDLED.items():
        out[k] = f.exists()
    if RAW.exists():
        for d in sorted(RAW.iterdir()):
            if (d / ".done").exists() and d.name not in ("stanford_lego_knights", "stanford_bracelet", "lytro_illum_flowers"):
                out[d.name.removeprefix("stanford_")] = True
    return out


# --------------------------------------------------------------------------
# user-supplied light fields (used by the app's upload option)
# --------------------------------------------------------------------------
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_GRID_RE = re.compile(r"(\d+)[_\-x](\d+)")


def _box_down(img: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return img
    H, W = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:H, :W].reshape(H // f, f, W // f, f, -1).mean(axis=(1, 3))


def _check_budget(n_values: int, max_mb: float):
    mb = n_values * 4 / 1e6
    if mb > max_mb:
        raise MemoryError(f"light field would need {mb:.0f} MB as float32 (limit {max_mb:.0f} MB); "
                          "increase the downsample factor or reduce the angular crop")


def from_array(arr: np.ndarray, name: str = "upload", downsample: int = 1,
               crop_angular: int | None = None) -> LightField:
    """Wrap an array of shape (U, V, H, W) or (U, V, H, W, C) as a LightField.
    Integer dtypes are scaled to [0, 1]; floats are divided by their maximum if
    it exceeds 1."""
    a = np.asarray(arr)
    if a.ndim not in (4, 5):
        raise ValueError(f"expected (U,V,H,W[,C]); got shape {a.shape}")
    a = _to_float(a)
    if a.max() > 1.0:
        a = a / a.max()
    out = LightField(a, name, {"source": "user array"})
    if crop_angular and crop_angular < min(out.nu, out.nv):
        out = out.crop_angular(crop_angular)
    return out.downsample(downsample) if downsample > 1 else out


def from_lenslet_image(img: np.ndarray, angular: tuple[int, int] = (14, 14), crop_angular: int | None = 9,
                       downsample: int = 1, gain: float | str | None = "auto", name: str = "upload",
                       max_mb: float = 700.0) -> LightField:
    """Decode a lenslet (ESLF-style) image: pixel (i, j) holds
    ``L[u = i % U, v = j % V, s = i // U, t = j // V]``.  The angular crop is
    applied on the integer data *before* converting to float, so a full Lytro
    Illum ESLF (5264 x 7574) decodes in well under 1 GB."""
    if img.ndim == 2:
        img = img[..., None]
    img = img[..., :3]
    au, av = angular
    H, W = (img.shape[0] // au) * au, (img.shape[1] // av) * av
    if H == 0 or W == 0:
        raise ValueError(f"image {img.shape[:2]} is smaller than one {au}x{av} micro-lens block")
    v = img[:H, :W].reshape(H // au, au, W // av, av, -1)           # (s, u, t, v, c) view, no copy
    if crop_angular and crop_angular < min(au, av):
        su, sv = au // 2 - crop_angular // 2, av // 2 - crop_angular // 2
        v = v[:, su:su + crop_angular, :, sv:sv + crop_angular]
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(v.dtype, 1.0)
    f = max(int(downsample), 1)
    S, T = (v.shape[0] // f) * f, (v.shape[2] // f) * f
    _check_budget(v.shape[1] * v.shape[3] * (S // f) * (T // f) * v.shape[4], max_mb)
    lfv = np.empty((v.shape[1], v.shape[3], S // f, T // f, v.shape[4]), np.float32)
    for u in range(v.shape[1]):               # one angular row at a time keeps the peak memory low
        row = v[:S, u, :T].transpose(2, 0, 1, 3).astype(np.float32)        # (v, s, t, c)
        if f > 1:                             # box-downsample while decoding
            row = row.reshape(row.shape[0], S // f, f, T // f, f, -1).mean(axis=(2, 4))
        lfv[u] = row
    del v, img
    lfv /= scale
    meta = {"source": "user lenslet image", "angular": angular}
    if gain is not None:
        c = lfv[lfv.shape[0] // 2, lfv.shape[1] // 2]
        g = float(1.0 / max(np.percentile(c, 99.5), 1e-6)) if gain == "auto" else float(gain)
        lfv *= g
        np.clip(lfv, 0, 1, out=lfv)
        meta["gain"] = g
    return LightField(lfv, name, meta)


def from_view_zip(data, downsample: int = 1, crop_angular: int | None = None, name: str = "upload",
                  max_mb: float = 700.0, max_members: int = 1200) -> LightField:
    """Build a light field from a ZIP of sub-aperture images (read in memory,
    nothing is extracted to disk).

    Accepted layouts, tried in this order:
    1. Stanford archive names ``out_RR_CC_<y>_<x>.png`` - axes are flipped as in
       :func:`load_stanford_rectified` so nearer objects get positive disparity.
    2. Any names whose last two integer groups are ``row`` and ``column``
       (``view_03_07.png``, ``3x7.jpg``, ...).
    3. Otherwise the files are sorted by name and must form a square grid
       (25 files -> 5 x 5, row-major)."""
    import io as _io, zipfile, math
    zf = zipfile.ZipFile(_io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    names = sorted(n for n in zf.namelist()
                   if n.lower().endswith(_IMG_EXT) and not pathlib.PurePosixPath(n).name.startswith("."))
    if not names:
        raise ValueError("no image files found in the ZIP")
    if len(names) > max_members:
        raise ValueError(f"{len(names)} images in the ZIP (limit {max_members})")
    ent, coords = {}, {}
    st = [(_STANFORD_RE.search(pathlib.PurePosixPath(n).name), n) for n in names]
    if all(m for m, _ in st):
        for m, n in st:
            ent[(int(m[1]), int(m[2]))] = n; coords[(int(m[1]), int(m[2]))] = (float(m[3]), float(m[4]))
    else:
        gm = [(_GRID_RE.findall(pathlib.PurePosixPath(n).stem), n) for n in names]
        if all(g for g, _ in gm) and len({tuple(map(int, g[-1])) for g, _ in gm}) == len(names):
            for g, n in gm:
                ent[tuple(map(int, g[-1]))] = n
        else:
            k = math.isqrt(len(names))
            if k * k != len(names):
                raise ValueError(f"{len(names)} images do not form a square grid and the names carry no row/column indices")
            for i, n in enumerate(names):
                ent[(i // k, i % k)] = n
    rows = sorted({r for r, _ in ent}); cols = sorted({c for _, c in ent})
    if len(rows) * len(cols) != len(ent):
        raise ValueError("view grid is incomplete (missing row/column combinations)")
    if crop_angular and crop_angular < min(len(rows), len(cols)):
        r0, c0 = len(rows) // 2 - crop_angular // 2, len(cols) // 2 - crop_angular // 2
        rows, cols = rows[r0:r0 + crop_angular], cols[c0:c0 + crop_angular]
    views = None
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            img = _to_float(iio.imread(zf.read(ent[(r, c)])))
            if img.ndim == 2:
                img = img[..., None]
            img = _box_down(img[..., :3], downsample)
            if views is None:
                _check_budget(len(rows) * len(cols) * img.size, max_mb)
                views = np.zeros((len(rows), len(cols)) + img.shape, np.float32)
            if img.shape != views.shape[2:]:
                raise ValueError(f"{ent[(r, c)]} has shape {img.shape}, expected {views.shape[2:]}")
            views[i, j] = img
    meta = {"source": "user ZIP of views", "downsample": downsample}
    if coords and len(rows) > 1 and len(cols) > 1:      # same orientation rule as the Stanford loader
        if coords[(rows[0], cols[-1])][1] > coords[(rows[0], cols[0])][1]:
            views = views[:, ::-1]
        if coords[(rows[-1], cols[0])][0] > coords[(rows[0], cols[0])][0]:
            views = views[::-1]
        meta["source"] = "user ZIP (Stanford naming)"
    return LightField(np.ascontiguousarray(views), name, meta)


def load_upload(data: bytes, filename: str, angular: tuple[int, int] = (14, 14),
                crop_angular: int | None = None, downsample: int = 1,
                gain: float | str | None = "auto") -> LightField:
    """Dispatch on the file extension: ``.zip`` (views), ``.npy`` / ``.npz``
    (array), or an image (lenslet / ESLF).  ``allow_pickle`` is off."""
    import io as _io
    ext = pathlib.PurePosixPath(filename).suffix.lower()
    stem = pathlib.PurePosixPath(filename).stem
    if ext == ".zip":
        return from_view_zip(data, downsample=downsample, crop_angular=crop_angular, name=stem)
    if ext in (".npy", ".npz"):
        a = np.load(_io.BytesIO(data), allow_pickle=False)
        if ext == ".npz":
            a = a[a.files[0]]
        return from_array(a, name=stem, downsample=downsample, crop_angular=crop_angular)
    if ext in _IMG_EXT:
        return from_lenslet_image(iio.imread(data), angular=angular, crop_angular=crop_angular,
                                  downsample=downsample, gain=gain, name=stem)
    raise ValueError(f"unsupported file type '{ext}' (use .zip, .npy, .npz or an image)")
