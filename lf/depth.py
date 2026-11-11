"""Disparity (depth) estimation from a 4-D light field.

Three estimators plus fusion and evaluation:

* :func:`depth_from_focus`          - defocus cue  (sharpest refocus, Tao et al. 2013)
* :func:`depth_from_correspondence` - correspondence cue (photo-consistency across views)
* :func:`epi_structure_tensor_disparity` - EPI line slope (Wanner & Goldluecke 2012)
* :func:`depth_combined`            - confidence-weighted fusion + guided-filter smoothing

All estimators return disparity in *our* convention (docs/conventions.md): the
pixel shift per unit angular step, ``d = 0`` on the rectification plane, nearer
points ``d > 0``.  Working in disparity (not metric depth) is deliberate, see
:func:`disparity_to_depth`.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter, binary_dilation
from .lightfield import LightField
from .utils import gray
from . import refocus as _rf

MEASURES = ("laplacian", "tenengrad", "variance")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gray_stack(stack):
    """(n,H,W,C) -> (n,H,W) luminance."""
    return np.stack([gray(s) for s in stack]).astype(np.float32)


def focus_measure(img, measure="laplacian", window=7):
    """Per-pixel, window-averaged sharpness of a gray image (H,W).

    laplacian : mean of (Laplacian)^2        (sum-modified-Laplacian flavour)
    tenengrad : mean of |Sobel gradient|^2
    variance  : local variance in the window
    """
    import cv2
    g = np.ascontiguousarray(img, dtype=np.float32)
    if measure == "laplacian":
        m = cv2.Laplacian(g, cv2.CV_32F, ksize=1) ** 2
    elif measure == "tenengrad":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        m = gx * gx + gy * gy
    elif measure == "variance":
        mu = uniform_filter(g, window)
        return np.maximum(uniform_filter(g * g, window) - mu * mu, 0).astype(np.float32)
    else:
        raise ValueError(f"measure must be one of {MEASURES}")
    return uniform_filter(m, window).astype(np.float32)


def _focus_volume(gstack, measure, window):
    return np.stack([focus_measure(g, measure, window) for g in gstack])


def _variance_volume(lf, alphas, window, return_stack=False):
    """Per-alpha angular variance of the sheared LF (window averaged)."""
    out, means = [], []
    for a in alphas:
        sh = _rf.shear(lf, float(a))
        var = sh.var(axis=(0, 1)).mean(axis=-1)
        out.append(uniform_filter(var, window))
        if return_stack:
            means.append(sh.mean(axis=(0, 1), dtype=np.float32))
    out = np.stack(out).astype(np.float32)
    return (out, np.stack(means)) if return_stack else out


def _argmax_refine(score, alphas, refine=True):
    """argmax over axis 0 of ``score`` (n,H,W), optional parabolic refinement.

    Fit a parabola through (i-1, i, i+1): delta = 0.5 (y- - y+) / (y- - 2 y0 + y+),
    clipped to [-0.5, 0.5]; the sub-sample index is mapped to alpha by
    linear interpolation (so non-uniform alpha grids work).
    """
    alphas = np.asarray(alphas, np.float64)
    n = score.shape[0]
    idx = np.argmax(score, axis=0)
    if not refine or n < 3:
        return alphas[idx].astype(np.float32), idx
    im = np.clip(idx - 1, 0, n - 1); ip = np.clip(idx + 1, 0, n - 1)
    take = lambda i: np.take_along_axis(score, i[None], 0)[0]
    y0, ym, yp = take(idx), take(im), take(ip)
    den = ym - 2 * y0 + yp
    delta = np.where(den < -1e-12, 0.5 * (ym - yp) / np.where(den < -1e-12, den, -1.0), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    delta[(idx == 0) | (idx == n - 1)] = 0.0
    d = np.interp(idx + delta, np.arange(n), alphas)
    return d.astype(np.float32), idx


def _prominence(score, floor_frac=0.0):
    """Peak prominence in [0,1] for a non-negative 'higher is better' score:
    (peak - mean) / peak.  1 = a single sharp peak, 0 = flat (no information)."""
    peak = score.max(0); mean = score.mean(0)
    c = (peak - mean) / (peak + 1e-12)
    return np.clip(c, 0, 1).astype(np.float32)


def _norm01(score):
    """Per-pixel min-max normalisation along alpha ('higher is better' -> [0,1])."""
    lo = score.min(0, keepdims=True); hi = score.max(0, keepdims=True)
    return ((score - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _focus_confidence(vol):
    # Prominence alone is high on noise in flat regions, so also demand that the
    # peak sharpness is not negligible relative to the image average.
    c = _prominence(vol)
    pk = vol.max(0)
    return (c * pk / (pk + 0.1 * pk.mean() + 1e-12)).astype(np.float32)


def _corr_confidence(var):
    # score = (mean variance - variance): large when one alpha is clearly consistent
    mean = var.mean(0); mn = var.min(0)
    return np.clip((mean - mn) / (mean + 1e-6), 0, 1).astype(np.float32)


def guided_filter(guide, src, radius=4, eps=1e-3):
    """Edge-preserving guided filter (He, Sun & Tang, ECCV 2010 / TPAMI 2013).

    ``q = mean(a) * I + mean(b)`` with ``a = cov(I,p)/(var(I)+eps)``,
    ``b = mean(p) - a mean(I)`` computed in (2r+1)^2 windows.  ``guide`` is a
    gray (H,W) image, ``src`` (H,W) or (n,H,W).
    """
    I = gray(np.asarray(guide, np.float32)).astype(np.float32)
    box = lambda x: uniform_filter(x, 2 * radius + 1, mode="nearest")
    mI = box(I); vI = box(I * I) - mI * mI

    def one(p):
        mp = box(p); cov = box(I * p) - mI * mp
        a = cov / (vI + eps); b = mp - a * mI
        return box(a) * I + box(b)
    src = np.asarray(src, np.float32)
    return (one(src) if src.ndim == 2 else np.stack([one(s) for s in src])).astype(np.float32)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def depth_from_focus(lf: LightField, alphas, measure="laplacian", window=7, refine=True) -> dict:
    """Depth from defocus: the alpha at which a pixel is sharpest.

    Build the focal stack ``I_alpha`` (shift-and-add), compute a window-averaged
    sharpness ``F(alpha, x)`` and take ``d(x) = argmax_alpha F`` with parabolic
    sub-sample refinement.  Fails on textureless regions (F is flat) but
    handles occlusion boundaries reasonably (Tao et al. 2013).

    Returns dict(disparity (H,W), confidence (H,W) in [0,1], cost_volume (n,H,W)
    = focus measure, higher is better).
    Reference: Tao, Hadap, Malik, Ramamoorthi, ICCV 2013.
    """
    stack = _rf.focal_stack(lf, alphas)
    vol = _focus_volume(_gray_stack(stack), measure, window)
    d, _ = _argmax_refine(vol, alphas, refine)
    return {"disparity": d, "confidence": _focus_confidence(vol), "cost_volume": vol}


def depth_from_correspondence(lf: LightField, alphas, window=5, refine=True) -> dict:
    """Depth from correspondence (photo-consistency).

    For each candidate alpha shear the light field and compute the variance
    across the U*V angular samples of every pixel.  On the true disparity all
    views see the same scene point, so the variance is minimal; this is the
    EPI-slope search made explicit.  Fails on textureless regions (variance is
    flat), repetitive texture (several minima) and at occlusions (views see
    different surfaces).

    Returns dict(disparity, confidence, cost_volume = window-averaged variance,
    lower is better).
    Reference: Tao et al. ICCV 2013.
    """
    var = _variance_volume(lf, alphas, window)
    d, _ = _argmax_refine(-var, alphas, refine)
    return {"disparity": d, "confidence": _corr_confidence(var), "cost_volume": var}


def epi_structure_tensor_disparity(lf: LightField, sigma=1.0, outer_sigma=2.0, max_disp=6.0) -> dict:
    """Disparity from the local line slope in EPIs via the 2-D structure tensor.

    Horizontal EPI at (u0, row s): ``E(v, t)``.  A scene point of disparity d
    traces the line ``t = t0 + d (v - v0)`` (docs/conventions.md), so ``E`` is
    constant along direction ``(dv, dt) = (1, d)``.  Its gradient ``(E_v, E_t)``
    is perpendicular to that line: ``E_v + d E_t = 0`` and therefore
    ``d = -E_v / E_t``.  With the structure tensor
    ``J = G_outer * [[E_v^2, E_v E_t], [E_v E_t, E_t^2]]`` the dominant
    eigenvector makes angle ``phi = 0.5 atan2(2 J_vt, J_vv - J_tt)`` with the v
    axis and ``d = -cot(phi)`` (total least squares).  The sign: a point on
    the *right* in the later views (v increasing) has a line leaning towards
    increasing t, i.e. positive slope = positive disparity = nearer than the
    focal plane.  The vertical EPI ``E(u, s)`` gives a second estimate with
    the same formula; the two are fused with weights
    ``coherence * trace / (trace + tiny)``.

    Parameters: ``sigma`` derivative scale (pixels / views), ``outer_sigma``
    spatial integration scale of the tensor, estimates are clipped to
    ``+-max_disp``.

    Returns dict(disparity, confidence (coherence-based, [0,1]),
    disparity_h, disparity_v).
    Reference: Wanner & Goldluecke, "Globally consistent depth labeling of 4D
    light fields", CVPR 2012.
    """
    G = np.stack([[gray(lf.views[u, v]) for v in range(lf.nv)] for u in range(lf.nu)]).astype(np.float32)
    u0, v0 = lf.center

    def estimate(slab, ang_axis, ang_idx, other_axis):
        # slab: (A, S, T) with A the angular axis; spatial axis for the line = axis 2
        ga = gaussian_filter(slab, sigma, order=(1, 0, 0), mode="nearest")[ang_idx]
        gs = gaussian_filter(slab, sigma, order=(0, 0, 1), mode="nearest")[ang_idx]
        Jaa = gaussian_filter(ga * ga, outer_sigma); Jss = gaussian_filter(gs * gs, outer_sigma)
        Jas = gaussian_filter(ga * gs, outer_sigma)
        phi = 0.5 * np.arctan2(2 * Jas, Jaa - Jss)
        s, c = np.sin(phi), np.cos(phi)
        s = np.where(np.abs(s) < 1e-6, 1e-6, s)
        d = np.clip(-c / s, -max_disp, max_disp)
        tr = Jaa + Jss
        coh = np.sqrt((Jaa - Jss) ** 2 + 4 * Jas ** 2) / (tr + 1e-12)
        w = coh * tr / (tr + 1e-4)
        return d.astype(np.float32), w.astype(np.float32)

    # horizontal: G[u0] is (V, H, W): angular v, spatial line axis t (axis 2)
    dh, wh = estimate(G[u0], 0, v0, 2)
    # vertical: G[:, v0] is (U, H, W); the line axis is s (axis 1) -> swap to (U, W, H)
    dv, wv = estimate(np.ascontiguousarray(G[:, v0].transpose(0, 2, 1)), 0, u0, 2)
    dv, wv = dv.T, wv.T
    w = wh + wv
    d = np.where(w > 1e-8, (wh * dh + wv * dv) / np.maximum(w, 1e-8), 0.0).astype(np.float32)
    return {"disparity": d, "confidence": np.clip(w / 2, 0, 1).astype(np.float32),
            "disparity_h": dh, "disparity_v": dv}


def depth_combined(lf: LightField, alphas, measure="laplacian", window_focus=7, window_corr=5,
                   refine=True, smooth=True, radius=5, eps=1e-3, median=3) -> dict:
    """Fuse defocus and correspondence, then smooth edge-aware.

    Design choice (a simplified Tao et al. 2013): both cost volumes are
    min-max normalised per pixel to a 'match score' in [0, 1]
    (1 = best alpha), then added with their own confidences as weights,
    ``S = c_def * S_def + c_cor * S_cor``; the fused disparity is the
    (parabola-refined) argmax of ``S``.  Where one cue is uninformative
    (flat cost curve, low confidence) the other dominates.  Tao uses an MRF
    for the final regularisation; we use *normalised convolution with a guided
    filter*: ``d = GF(c d) / GF(c)``, guided by the centre view so depth edges
    follow image edges, followed by a small median to remove outliers.

    Returns dict(disparity (smoothed), disparity_raw, confidence,
    disparity_defocus, disparity_corr, confidence_defocus, confidence_corr, score).
    Reference: Tao et al. ICCV 2013; He et al. (guided filter) TPAMI 2013.
    """
    var, means = _variance_volume(lf, alphas, window_corr, return_stack=True)
    fvol = _focus_volume(_gray_stack(means), measure, window_focus)
    d_f, _ = _argmax_refine(fvol, alphas, refine)
    d_c, _ = _argmax_refine(-var, alphas, refine)
    c_f, c_c = _focus_confidence(fvol), _corr_confidence(var)
    S = c_f[None] * _norm01(fvol) + c_c[None] * _norm01(-var)
    d_raw, _ = _argmax_refine(S, alphas, refine)
    conf = np.clip(0.5 * (c_f + c_c), 0, 1).astype(np.float32)
    d = d_raw
    if smooth:
        guide = gray(lf.center_view)
        w = conf + 1e-3
        num = guided_filter(guide, d_raw * w, radius, eps)
        den = guided_filter(guide, w, radius, eps)
        d = np.where(den > 1e-6, num / np.maximum(den, 1e-6), d_raw)
        if median and median > 1:
            d = median_filter(d, size=median, mode="nearest")
        d = d.astype(np.float32)
    return {"disparity": d, "disparity_raw": d_raw, "confidence": conf,
            "disparity_defocus": d_f, "disparity_corr": d_c,
            "confidence_defocus": c_f, "confidence_corr": c_c, "score": S}


# ---------------------------------------------------------------------------
# evaluation / conversion
# ---------------------------------------------------------------------------
def discontinuity_mask(gt, band=3, thresh=0.25):
    """Boolean map, True within ``band`` pixels of a depth discontinuity of gt."""
    gy, gx = np.gradient(np.asarray(gt, np.float32))
    edge = np.hypot(gx, gy) > thresh
    return binary_dilation(edge, iterations=band) if band > 0 else edge


def evaluate(disparity, gt, mask=None, band=3, bad_thresh=0.25, border=0) -> dict:
    """Error metrics of a disparity map.

    ``bad = |err| > bad_thresh`` (disparity units).  By default a ``band`` of 3 px
    around gt depth discontinuities is excluded: at an edge a window straddles
    two surfaces and the 'correct' value is ambiguous, and the gt itself
    is only defined up to the sub-pixel position of the edge.  Metrics over that
    band mostly measure window size rather than algorithm quality.  The all-pixel
    numbers are reported too (keys ``*_all``).  ``mask`` (True = evaluate) is
    ANDed with the discontinuity exclusion; ``border`` also drops an image margin.

    Returns dict(mae, rmse, bad_pixel_ratio, mae_all, rmse_all, bad_pixel_ratio_all, n_pixels).
    """
    d = np.asarray(disparity, np.float64); g = np.asarray(gt, np.float64)
    err = d - g
    valid_all = np.ones_like(g, bool) if mask is None else np.asarray(mask, bool).copy()
    if border > 0:
        b = np.zeros_like(valid_all); b[border:-border, border:-border] = True
        valid_all &= b
    valid = valid_all & ~discontinuity_mask(g, band) if band > 0 else valid_all

    def stats(m):
        if not m.any():
            return 0.0, 0.0, 0.0
        e = err[m]
        return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), float((np.abs(e) > bad_thresh).mean())
    mae, rmse, bad = stats(valid); mae_a, rmse_a, bad_a = stats(valid_all)
    return {"mae": mae, "rmse": rmse, "bad_pixel_ratio": bad, "mae_all": mae_a, "rmse_all": rmse_a,
            "bad_pixel_ratio_all": bad_a, "n_pixels": int(valid.sum())}


def disparity_to_depth(disparity, baseline, focal_px, d0_depth=None):
    """Convert disparity (pixels per angular step) to metric depth ``z``.

    For a camera translating by ``baseline`` between adjacent angular samples,
    a point at depth ``z`` has absolute disparity ``f b / z`` (``f`` in pixels).
    Our ``d`` is measured relative to the rectification plane at depth ``z0``:
    ``d = f b (1/z - 1/z0)``, hence ``z = f b / (d + f b / z0)``.  With
    ``d0_depth=None`` we take ``z0 = inf`` (``d`` absolute, ``z = f b / d``).
    ``z`` is proportional to ``1/(d + const)``: depth is nonlinear in d, noise
    in d explodes at large z, and the unknown constant (z0) is why we estimate
    and compare in disparity.  Non-positive denominators map to ``inf``.
    """
    d = np.asarray(disparity, np.float64)
    den = d + (0.0 if d0_depth is None else focal_px * baseline / d0_depth)
    with np.errstate(divide="ignore"):
        z = np.where(den > 1e-9, focal_px * baseline / np.maximum(den, 1e-9), np.inf)
    return z.astype(np.float32)
