"""Digital refocusing by shift-and-add (Ng et al. 2005, lecture 15 slides 45-56).

Convention (docs/conventions.md): refocusing on the plane of disparity ``alpha``
means shifting view ``(u, v)`` by ``(-alpha (u-u0), -alpha (v-v0))`` pixels and
averaging.  See ``docs/03_refocusing.md`` for the derivation.

Boundary handling
-----------------
* ``'subpixel'`` / ``'integer'`` : border pixels are replicated (``cv2.BORDER_REPLICATE``
  / ``mode='nearest'``), so edges are smeared but never wrap around.
* ``'fourier'`` : the shift is a phase ramp, hence *periodic* -- content that
  leaves one side re-enters on the other.  Exact for band-limited signals, but
  can create wrap-around ghosts at the image border for large |alpha|.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import shift as nd_shift, laplace
from .lightfield import LightField

# Disparity units of the *loaded* LF (pixels per angular step).
# 'synthetic' : synthetic_light_field(n_views=7..9, size=96..256), layers at -1, 0, +1.5.
# 'lego'      : lf.io.load('lego', downsample=4, angular_crop=9) -> 256x256, 9x9 views.
#               Filled in by measurement with sweep_sharpness (see docs/03_refocusing.md).
# 'bracelet'  : not measured (dataset not available when this was written).
SUGGESTED_ALPHA_RANGE = {
    "synthetic": (-1.5, 2.0),
    "lego": (-2.0, 1.5),      # lf.io.load('lego', downsample=4, angular_crop=9): wall -1.0, knights 0, near floor +1.0
    "lytro": (-1.5, 1.0),     # lf.io.load('lytro') (crop 9, full res): blossoms -0.5..+0.3, sky/background to -1.5
    "bracelet": (-1.0, 2.0),  # lf.io.load('bracelet', downsample=4, angular_crop=9): far clasp 0, near chain +1.0..+1.5
}


# ----------------------------------------------------------------------------
# Ng's parameterisation
# ----------------------------------------------------------------------------
def ng_alpha_to_disparity(alpha_ng, pixels_per_aperture_step: float = 1.0):
    """Convert Ng's refocus parameter ``alpha' = F'/F`` to our shift slope ``alpha``.

    Ng moves the virtual film plane from depth F to F' = alpha' F.  Using the
    two-plane light field L(u, v, s, t) (u on the lens, s on the film at depth F)
    a ray hitting the new film at s passes the lens at u and hits the old film at
        s_old = u + (s - u) / alpha'  =  s / alpha' + (1 - 1/alpha') u .
    Hence  E_{alpha' F}(s) ~ sum_u L(u, s/alpha' + (1 - 1/alpha') u).
    Apart from a global magnification 1/alpha' (a pure image rescale that does
    not change which depth is sharp), each sub-aperture image u is sampled at
    ``s + (1 - 1/alpha') u``: a shift of slope (1 - 1/alpha') per unit of u.
    If one angular step of our grid equals ``pixels_per_aperture_step`` sensor
    pixels of u, then  ``alpha = (1 - 1/alpha') * pixels_per_aperture_step``.
    alpha' = 1 gives 0 (focus on the original plane); alpha' > 1 (film moved
    away from the lens) gives positive alpha, i.e. nearer scene planes.

    Reference: Ng et al., Stanford TR 2005-02, Sec. 4 (eq. 4-5).
    """
    a = np.asarray(alpha_ng, dtype=float)
    return (1.0 - 1.0 / a) * pixels_per_aperture_step


def disparity_to_ng_alpha(alpha, pixels_per_aperture_step: float = 1.0):
    """Inverse of :func:`ng_alpha_to_disparity`: ``alpha' = 1 / (1 - alpha / p)``."""
    d = np.asarray(alpha, dtype=float)
    return 1.0 / (1.0 - d / pixels_per_aperture_step)


# ----------------------------------------------------------------------------
# Shifting kernels
# ----------------------------------------------------------------------------
def _shift_cv(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Bilinear shift of (H, W, C) by (dy, dx) with replicated borders (cv2)."""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    out = cv2.warpAffine(np.ascontiguousarray(img), M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(img.shape)  # cv2 drops a trailing channel axis of size 1


def shear(lf: LightField, alpha: float, order: int = 1, mode: str = "nearest") -> np.ndarray:
    """Return the sheared light field ``L'(u,v,s,t) = L(u,v, s+alpha(u-u0), t+alpha(v-v0))``
    as an array of shape (U, V, H, W, C).  After shearing by the disparity of a
    scene plane, that plane is aligned across all views (zero disparity)."""
    U, V = lf.nu, lf.nv
    u0, v0 = lf.center
    out = np.empty_like(lf.views)
    for u in range(U):
        for v in range(V):
            dy, dx = -alpha * (u - u0), -alpha * (v - v0)
            if dy == 0 and dx == 0:
                out[u, v] = lf.views[u, v]
            else:
                out[u, v] = nd_shift(lf.views[u, v], (dy, dx, 0), order=order, mode=mode)
    return out


def refocus(lf: LightField, alpha: float, method: str = "subpixel",
            mask: np.ndarray | None = None) -> np.ndarray:
    """Photograph focused on the plane of disparity ``alpha`` (shape (H, W, C)).

    ``I_alpha(s,t) = sum_{u,v} m(u,v) L(u,v, s+alpha(u-u0), t+alpha(v-v0)) / sum m``

    Parameters
    ----------
    alpha : disparity (pixels per angular step) of the plane to bring into focus.
    method : 'subpixel' (bilinear, cv2, replicated borders), 'integer' (each
        view's shift rounded to whole pixels, replicated borders) or 'fourier'
        (phase-ramp shift, exact for band-limited images, periodic borders).
    mask : optional (U, V) aperture weights (digital stopping-down); ``None``
        is identical to an all-ones mask.  Views with zero weight are skipped.

    Reference: Ng, Levoy, Bredif, Duval, Horowitz, Hanrahan, "Light field
    photography with a hand-held plenoptic camera", Stanford TR 2005-02.
    """
    if method not in ("subpixel", "integer", "fourier"):
        raise ValueError(f"unknown method {method!r}")
    U, V = lf.nu, lf.nv
    u0, v0 = lf.center
    m = np.ones((U, V), np.float32) if mask is None else np.asarray(mask, np.float32)
    if m.shape != (U, V):
        raise ValueError(f"mask shape {m.shape} != {(U, V)}")
    msum = float(m.sum())
    if msum <= 0:
        raise ValueError("mask has zero total weight")
    H, W = lf.height, lf.width

    if method == "fourier":
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.rfftfreq(W)[None, :]
        acc = 0
        for u in range(U):
            for v in range(V):
                if m[u, v] == 0:
                    continue
                dy, dx = -alpha * (u - u0), -alpha * (v - v0)
                F = np.fft.rfft2(lf.views[u, v], axes=(0, 1))
                ramp = np.exp(-2j * np.pi * (fy * dy + fx * dx)).astype(np.complex64)
                acc = acc + (m[u, v] * ramp)[..., None] * F
        return (np.fft.irfft2(acc, s=(H, W), axes=(0, 1)) / msum).astype(np.float32)

    acc = np.zeros(lf.views.shape[2:], np.float32)
    for u in range(U):
        for v in range(V):
            if m[u, v] == 0:
                continue
            dy, dx = -alpha * (u - u0), -alpha * (v - v0)
            if method == "integer":
                dy, dx = float(np.round(dy)), float(np.round(dx))
            img = lf.views[u, v] if (dy == 0 and dx == 0) else _shift_cv(lf.views[u, v], dy, dx)
            acc += m[u, v] * img
    return acc / msum


def focal_stack(lf: LightField, alphas, **kw) -> np.ndarray:
    """Stack of refocused images for each alpha; shape (len(alphas), H, W, C)."""
    return np.stack([refocus(lf, float(a), **kw) for a in alphas]).astype(np.float32)


# ----------------------------------------------------------------------------
# Sharpness
# ----------------------------------------------------------------------------
def sharpness(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian of the grey image (higher = sharper).

    ``mask`` is an optional boolean (H, W) region of interest.  The outermost
    2 pixels are ignored to avoid border artefacts."""
    g = np.asarray(img, np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    lap = laplace(g)
    valid = np.zeros(g.shape, bool)
    valid[2:-2, 2:-2] = True
    if mask is not None:
        valid &= np.asarray(mask, bool)
    return float(lap[valid].var()) if valid.any() else 0.0


def sweep_sharpness(lf: LightField, alphas, roi=None, **kw) -> np.ndarray:
    """Sharpness of the refocused image for each alpha.

    ``roi`` is either an (H, W) boolean mask or a slice pair ``(slice_rows,
    slice_cols)``.  Returns an array of len(alphas)."""
    mask = None
    if roi is not None:
        if isinstance(roi, tuple) and isinstance(roi[0], slice):
            mask = np.zeros((lf.height, lf.width), bool)
            mask[roi] = True
        else:
            mask = np.asarray(roi, bool)
    return np.array([sharpness(refocus(lf, float(a), **kw), mask) for a in alphas])
