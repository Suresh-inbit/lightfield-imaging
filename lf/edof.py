"""Extended depth of field (all-in-focus) from a light field.

Slide 58-59 of lecture 15: a conventional photo at f/22 is sharp everywhere
but dark/noisy, at f/4 it is bright but has shallow DoF.  A light field
captured at f/4 can be refocused to every depth and the sharp parts merged.
Agarwala et al. (2004) select, per pixel, the best source image with a graph
cut (MRF); here we use a cheaper local method: soft per-pixel sharpness
weights, smoothed (guided / Gaussian filter) and normalised.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .lightfield import LightField
from . import refocus as _rf
from .depth import _gray_stack, _focus_volume, guided_filter, MEASURES
from .utils import gray


def all_in_focus(lf: LightField, alphas, measure="laplacian", window=7, smoothing="guided",
                 sigma=2.0, radius=3, eps=1e-3, beta=20.0) -> dict:
    """Merge a focal stack into one sharp image.

    1. focal stack ``I_k = refocus(alpha_k)``;
    2. sharpness ``F_k(x)`` (window-averaged ``measure``);
    3. normalise per pixel, ``z_k = F_k / max_k F_k``, and weights
       ``w_k = exp(beta (z_k - 1))`` (a soft arg-max, beta = peakedness);
    4. smooth each weight map: ``'guided'`` (guided filter, radius, eps; guide =
       centre view, so selection boundaries follow image edges), ``'gaussian'``
       (std ``sigma``) or ``'none'`` (hard arg-max, one-hot weights);
    5. ``image = sum_k w_k I_k / sum_k w_k``.

    Agarwala et al. 2004 solve step 3-4 with a graph cut; this is a local
    approximation and can leave halos at depth edges.

    Returns dict(image (H,W,C), index_map (H,W) int into ``alphas``,
    disparity (H,W) = alphas[index_map], focus_measure (n,H,W)).
    Reference: Agarwala et al., "Interactive digital photomontage", SIGGRAPH 2004.
    """
    alphas = np.asarray(alphas, np.float64)
    stack = _rf.focal_stack(lf, alphas)
    F = _focus_volume(_gray_stack(stack), measure, window)
    z = F / (F.max(0, keepdims=True) + 1e-12)
    if smoothing == "none":
        w = (np.arange(len(alphas))[:, None, None] == F.argmax(0)[None]).astype(np.float32)
    else:
        w = np.exp(beta * (z - 1)).astype(np.float32)
        if smoothing == "gaussian":
            w = np.stack([gaussian_filter(x, sigma) for x in w])
        elif smoothing == "guided":
            w = np.maximum(guided_filter(gray(lf.center_view), w, radius, eps), 0)
        else:
            raise ValueError("smoothing must be 'guided', 'gaussian' or 'none'")
    w = w / (w.sum(0, keepdims=True) + 1e-12)
    img = np.einsum('nhw,nhwc->hwc', w, stack)
    idx = w.argmax(0).astype(np.int64)
    return {"image": np.clip(img, 0, 1).astype(np.float32), "index_map": idx,
            "disparity": alphas[idx].astype(np.float32), "focus_measure": F}


def compare_with_stop_down(lf: LightField, alphas, small_radius=1, noise_sigma=None,
                           focus_alpha=0.0, seed=0, **aif_kw):
    """Slide-58 triptych: (full aperture at ``focus_alpha``, stopped down, all-in-focus).

    The stopped-down image uses only angular samples within ``small_radius`` of
    the centre (``lf.aperture.stop_down`` when available, else a circular mask
    in ``lf.refocus.refocus``) refocused at the same alpha: small aperture ->
    large DoF.  Stopping down also throws away light.  If ``noise_sigma`` is
    given it is the (post-exposure-matching) noise std of the full aperture
    capture; the stopped-down image gets Gaussian noise with std
    ``noise_sigma * sqrt(A_full / A_small)`` (shot-noise-like: collecting
    ``A_small / A_full`` of the photons lowers SNR by sqrt of that ratio),
    clipped to [0, 1].  The full-aperture and all-in-focus images are
    left noise-free (they use the full light budget).

    Returns (full, stopped, all_in_focus), each (H,W,C).
    """
    full = _rf.refocus(lf, focus_alpha)
    u0, v0 = lf.center
    uu, vv = np.meshgrid(np.arange(lf.nu) - u0, np.arange(lf.nv) - v0, indexing="ij")
    mask = ((uu ** 2 + vv ** 2) <= small_radius ** 2 + 1e-6).astype(np.float32)
    try:                                   # lazy: lf.aperture may not exist yet
        from . import aperture as _ap
        stopped = _ap.stop_down(lf, small_radius, alpha=focus_alpha)
    except Exception:
        stopped = _rf.refocus(lf, focus_alpha, mask=mask)
    stopped = np.asarray(stopped, np.float32)
    if noise_sigma:
        ratio = (lf.nu * lf.nv) / max(float(mask.sum()), 1.0)
        rng = np.random.default_rng(seed)
        stopped = np.clip(stopped + rng.normal(0, noise_sigma * np.sqrt(ratio), stopped.shape), 0, 1).astype(np.float32)
    aif = all_in_focus(lf, alphas, **aif_kw)["image"]
    return full, stopped, aif
