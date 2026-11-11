"""Plenoptic (lenslet) sensor image: simulate it, decode it, reason about its design.

Lecture 15 (Ng et al. 2005), slides 29-43.  A plenoptic camera puts a micro-lens
array on the sensor plane ``s``; each micro-lens ``(s, t)`` covers a ``U x V``
block of pixels, and pixel ``(i, j)`` of that block records the ray that came
through aperture position ``(u, v)`` (the micro-lens images the main-lens
aperture onto its own pixels).  Hence the raw image is the 4-D light field
re-arranged ("interleaved") into a mosaic of ``U x V`` tiles::

    raw[s*U + i, t*V + j] = L[u(i), v(j), s, t]

Orientation
-----------
A real micro-lens forms an *inverted* image of the aperture, so pixel offset
``(i, j)`` sees aperture position ``(U-1-i, V-1-j)``.  ``invert=True`` applies
that flip.  The default ``invert=False`` is the Stanford ESLF convention used
by :func:`lf.io.load_lytro_eslf`: ``u = i % U``, ``v = j % V``.

Raw images are ``(U*H, V*W, C)`` float32 arrays.
"""
from __future__ import annotations
import numpy as np
from .lightfield import LightField


# ----------------------------------------------------------------------------
# Raw <-> light field
# ----------------------------------------------------------------------------
def to_lenslet(lf: LightField, invert: bool = False) -> np.ndarray:
    """Simulate the raw lenslet mosaic, shape ``(U*H, V*W, C)``.

    The ``U x V`` pixels under micro-lens ``(s, t)`` hold ``L[:, :, s, t]``
    (flipped in both angular axes if ``invert``, as in a real camera)."""
    v = lf.views[::-1, ::-1] if invert else lf.views
    U, V, H, W, C = v.shape
    # (u, v, s, t, c) -> (s, u, t, v, c) -> (s*U + u, t*V + v, c)
    return np.ascontiguousarray(v.transpose(2, 0, 3, 1, 4).reshape(H * U, W * V, C))


def from_lenslet(raw: np.ndarray, angular=(14, 14), invert: bool = False,
                 name: str = "lenslet") -> LightField:
    """Inverse of :func:`to_lenslet`: take the same pixel under every micro-lens
    to build each sub-aperture image (slide 41).  Trailing rows/columns that do
    not fill a whole micro-lens are dropped."""
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = raw[..., None]
    U, V = angular
    H, W = raw.shape[0] // U, raw.shape[1] // V
    v = raw[:H * U, :W * V].reshape(H, U, W, V, -1).transpose(1, 3, 0, 2, 4)
    if invert:
        v = v[::-1, ::-1]
    return LightField(np.ascontiguousarray(v), name, {"angular": tuple(angular), "invert": invert})


def lenslet_closeup(raw: np.ndarray, angular, center_st, n_lenses: int = 6) -> np.ndarray:
    """Crop of ``n_lenses x n_lenses`` micro-lens tiles around micro-lens
    ``center_st = (s, t)`` (slides 39-40).  The window is clamped to the image."""
    U, V = angular
    H, W = raw.shape[0] // U, raw.shape[1] // V
    s0 = int(np.clip(center_st[0] - n_lenses // 2, 0, max(H - n_lenses, 0)))
    t0 = int(np.clip(center_st[1] - n_lenses // 2, 0, max(W - n_lenses, 0)))
    return raw[s0 * U:(s0 + n_lenses) * U, t0 * V:(t0 + n_lenses) * V]


def disc_mask(nu: int, nv: int, radius: float | None = None) -> np.ndarray:
    """(U, V) boolean disc of pixels around ``(nu//2, nv//2)`` (same convention
    as ``LightField.center``); ``radius=None`` keeps every pixel."""
    if radius is None:
        return np.ones((nu, nv), bool)
    uu, vv = np.mgrid[0:nu, 0:nv]
    return np.hypot(uu - nu // 2, vv - nv // 2) <= radius + 1e-6


def conventional_photo(lf_or_raw, angular=None, radius: float | None = None,
                       invert: bool = False) -> np.ndarray:
    """The photo a *conventional* camera would take: average of all pixels under
    each micro-lens -> ``(H, W, C)``.  With ``radius`` (in pixels of the U x V
    block) only the central disc is used, i.e. digital stopping-down (slide 43).
    Equals ``refocus(lf, 0.0, mask=disc_mask(...))``."""
    lf = lf_or_raw if isinstance(lf_or_raw, LightField) else from_lenslet(lf_or_raw, angular or (14, 14), invert)
    m = disc_mask(lf.nu, lf.nv, radius).astype(np.float32)
    return (np.tensordot(m, lf.views, axes=([0, 1], [0, 1])) / m.sum()).astype(np.float32)


# ----------------------------------------------------------------------------
# Optical design (slides 31-37)
# ----------------------------------------------------------------------------
def _disc_square_overlap(d: float, p: float, n: int = 400) -> tuple[float, float]:
    """Numerically: (fraction of the disc of diameter d that lies outside the
    p x p cell centred on it, fraction of the cell not covered by the disc)."""
    g = (np.arange(n) + 0.5) / n - 0.5            # cell coordinates in units of p
    x, y = np.meshgrid(g * 1.0, g * 1.0)
    inside_cell = (np.hypot(x, y) <= d / (2 * p)).mean()   # disc area inside cell / p^2
    disc_area = np.pi * (d / p) ** 2 / 4
    outside = max(disc_area - inside_cell, 0.0) / disc_area
    return float(outside), float(1.0 - inside_cell)


def design(sensor_px=(4000, 4000), pixel_pitch_um: float = 9.0, microlens_pitch_um: float = 125.0,
           microlens_focal_um: float = 500.0, main_f_number: float = 4.0,
           n_microlenses=None) -> dict:
    """Numbers of a plenoptic design (Ng 2005, slides 31-37).

    * angular resolution = pixels per micro-lens = round(pitch / pixel);
    * spatial resolution = number of micro-lenses = floor(sensor * pixel / pitch),
      or ``n_microlenses`` if given (Ng's prototype quotes 292 x 292, which
      implies a pitch of 4000*9/292 = 123 um, "~125 um" on the slide);
    * micro-lens f-number = f_m / pitch; the main-lens f-number must equal it;
    * micro-lens image (of the aperture) has diameter d = f_m / N_main;
    * wasted = fraction of the p x p cell not lit by the disc; overlap = fraction
      of the disc that falls on neighbouring cells (cross-talk).  Both are
      geometric estimates for a circular aperture; with a perfect match the disc
      is inscribed in the cell, so ~21 % of the cell is still unlit.
    """
    sh, sw = sensor_px
    if n_microlenses is not None:
        ns, nt = n_microlenses
        pitch_px = sh / ns
        microlens_pitch_um = pitch_px * pixel_pitch_um
    else:
        pitch_px = microlens_pitch_um / pixel_pitch_um
        ns, nt = int(sh // pitch_px), int(sw // pitch_px)
    ang = int(round(pitch_px))
    n_micro = microlens_focal_um / microlens_pitch_um
    d_um = microlens_focal_um / main_f_number
    overlap, wasted = _disc_square_overlap(d_um, microlens_pitch_um)
    return {
        "angular_resolution": (ang, ang),
        "pixels_per_microlens": pitch_px,
        "n_microlenses": (ns, nt),
        "spatial_resolution": (ns, nt),
        "microlens_f_number": n_micro,
        "main_f_number": main_f_number,
        "f_numbers_match": bool(abs(n_micro - main_f_number) / main_f_number < 0.05),
        "microlens_image_diameter_um": d_um,
        "microlens_image_diameter_px": d_um / pixel_pitch_um,
        "overlap_fraction": overlap,
        "wasted_fraction": wasted,
        "total_rays": ns * nt * ang * ang,
        "sensor_pixels": sh * sw,
    }


def resolution_tradeoff(sensor_px, angular) -> tuple[int, int]:
    """Spatial resolution = M / N for a sensor of ``sensor_px = (Mh, Mw)`` pixels
    and ``angular = (Nu, Nv)`` angular samples (lecture 16, slide 14)."""
    return int(sensor_px[0] // angular[0]), int(sensor_px[1] // angular[1])


def camera_table() -> list[dict]:
    """Rows of the plenoptic trade-off table (lecture 16, slide 14)."""
    rows = [("Ng 2005 prototype (slide labels it Lytro, 16 MP)", (4000, 4000), (14, 14), (292, 292)),
            ("Sensor 2656x1216", (2656, 1216), (16, 16), (166, 76)),
            ("Sensor 4044x3240", (4044, 3240), (5, 4), (808, 810)),
            ("Sensor 1629x2052", (1629, 2052), (9, 9), (181, 228))]
    return [{"camera": n, "sensor_px": s, "angular": a, "spatial_slide": sp,
             "spatial_computed": resolution_tradeoff(s, a)} for n, s, a, sp in rows]


# ----------------------------------------------------------------------------
# f-number mismatch (slide 35)
# ----------------------------------------------------------------------------
def simulate_fnumber_mismatch(lf: LightField, main_f_number: float,
                              microlens_f_number: float = 4.0) -> np.ndarray:
    """Raw mosaic when the main lens is not matched to the micro-lenses.

    The aperture image under each micro-lens has diameter ``r`` times the cell,
    ``r = N_micro / N_main`` (f/2.8 main lens -> r = 1.43 ; f/8 -> r = 0.5).
    The ``U x V`` angular patch (masked to a disc) is rescaled by ``r`` about
    the tile centre (nearest-neighbour) and written into the mosaic.
    ``r < 1``: a small disc, unused pixels around it.  ``r > 1``: the disc spills
    into neighbouring tiles and overlapping contributions are *averaged*
    (cross-talk), so each pixel mixes rays of several micro-lenses.
    """
    r = microlens_f_number / main_f_number
    U, V, H, W, C = lf.views.shape
    cu, cv = (U - 1) / 2, (V - 1) / 2
    pad_u, pad_v = int(np.ceil(max(r - 1, 0) * U / 2)) + 1, int(np.ceil(max(r - 1, 0) * V / 2)) + 1
    acc = np.zeros((H * U + 2 * pad_u, W * V + 2 * pad_v, C), np.float32)
    cnt = np.zeros(acc.shape[:2], np.float32)
    for a in range(-pad_u, U + pad_u):
        for b in range(-pad_v, V + pad_v):
            su, sv = cu + (a - cu) / r, cv + (b - cv) / r        # source angular position
            iu, iv = int(round(su)), int(round(sv))
            if not (0 <= iu < U and 0 <= iv < V) or np.hypot(su - cu, sv - cv) > max(U, V) / 2:
                continue
            ys = np.arange(H) * U + a + pad_u
            xs = np.arange(W) * V + b + pad_v
            acc[np.ix_(ys, xs)] += lf.views[iu, iv].transpose(0, 1, 2)
            cnt[np.ix_(ys, xs)] += 1
    out = acc / np.maximum(cnt, 1)[..., None]
    return out[pad_u:pad_u + H * U, pad_v:pad_v + W * V]
