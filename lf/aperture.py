"""Aperture control and viewpoint manipulation (Ng et al. 2005; slides 41, 43, 56, 61-64).

All functions work on the (U, V) angular grid of a :class:`lf.LightField`.
Radii are measured in angular steps (view spacings) from ``center``.
"""
from __future__ import annotations
import numpy as np
import cv2
from .lightfield import LightField
from .refocus import refocus


def aperture_mask(nu: int, nv: int, shape: str = "circle", radius: float | None = None,
                  center: tuple[float, float] | None = None, soft: bool = False) -> np.ndarray:
    """(U, V) float32 aperture weights.

    shape : 'circle' (distance <= radius), 'square' (Chebyshev distance <=
        radius) or 'annulus' (radius/2 < distance <= radius).
    radius : in angular steps; default = covers the whole grid.
    center : (u, v), default grid centre.
    soft : if True use a smooth (cosine) edge one step wide instead of a hard cut.
    """
    if center is None:
        center = (nu // 2, nv // 2)
    if radius is None:
        radius = max(nu, nv) / 2.0
    uu, vv = np.mgrid[0:nu, 0:nv].astype(np.float32)
    du, dv = uu - center[0], vv - center[1]
    if shape == "square":
        d = np.maximum(np.abs(du), np.abs(dv))
    elif shape in ("circle", "annulus"):
        d = np.hypot(du, dv)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    def inside(r):
        if soft:
            return np.clip(0.5 - (d - r), 0.0, 1.0)  # linear ramp over one step around r
        return (d <= r + 1e-6).astype(np.float32)

    m = inside(radius)
    if shape == "annulus":
        m = m * (1.0 - inside(radius / 2.0))
    return m.astype(np.float32)


def stop_down(lf: LightField, radius: float, alpha: float = 0.0, shape: str = "circle",
              center=None, method: str = "subpixel") -> np.ndarray:
    """Digitally stopped-down photograph (slide 43): refocus at ``alpha`` using only
    the views inside the aperture of the given ``radius``.  Smaller radius ->
    larger depth of field but less light (more noise); radius 0 = one view.
    Reference: Ng et al., TR 2005-02, Sec. 6."""
    m = aperture_mask(lf.nu, lf.nv, shape, radius, center)
    return refocus(lf, alpha, method=method, mask=m)


def effective_f_number(f_main: float, pixels_per_lens: int, radius_in_pixels: float) -> float:
    """f-number of a digitally stopped-down LF image (digital refocusing theorem, slide 56).

    A main lens of f-number N and P x P pixels per microlens: the full aperture
    covers P pixels (diameter) -> f/N; a window of diameter w pixels lets
    through w/P of the aperture -> f/(N P / w).  One pixel (w = 1) gives f/(N P),
    the sharpest conventional-camera equivalent.  Here w = 2 r + 1 (r = radius
    in pixels, so radius 0 = single view), clipped to [1, P].
    """
    w = float(np.clip(2.0 * radius_in_pixels + 1.0, 1.0, pixels_per_lens))
    return f_main * pixels_per_lens / w


def synthesize_view(lf: LightField, u: float, v: float) -> np.ndarray:
    """Image seen from fractional angular position (u, v) by bilinear blending
    of the 4 nearest views.

    This is a light-field *interpolation* in (u, v): points of disparity d appear
    twice (at the positions of the two blended views) separated by d * frac
    pixels, so for |d| of more than a pixel or so you get ghosting / doubled
    edges.  Fixing that requires depth: warp each neighbour by d(s,t)*(du,dv)
    using an estimated disparity map before blending (depth-image-based rendering).
    """
    u = float(np.clip(u, 0, lf.nu - 1)); v = float(np.clip(v, 0, lf.nv - 1))
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = min(u0 + 1, lf.nu - 1), min(v0 + 1, lf.nv - 1)
    a, b = u - u0, v - v0
    X = lf.views
    return ((1 - a) * (1 - b) * X[u0, v0] + (1 - a) * b * X[u0, v1]
            + a * (1 - b) * X[u1, v0] + a * b * X[u1, v1]).astype(np.float32)


def perspective_sequence(lf: LightField, path: str = "circle", n: int = 24,
                         radius: float | None = None) -> list[np.ndarray]:
    """List of ``n`` images along a camera path (slides 61-62, moving the observer).

    path : 'circle' (around the centre), 'row' (left-right through the centre),
        'column' (up-down) or 'diagonal'.  ``radius`` in angular steps; default
        = the largest that fits in the grid.  Positions are fractional and use
        :func:`synthesize_view`.
    """
    u0, v0 = lf.center
    r = radius if radius is not None else min(lf.nu - 1 - u0, u0, lf.nv - 1 - v0, v0)
    out = []
    for k in range(n):
        if path == "circle":
            th = 2 * np.pi * k / n
            du, dv = r * np.sin(th), r * np.cos(th)
        else:
            s = r * np.sin(2 * np.pi * k / n)   # back-and-forth, loops seamlessly
            du, dv = {"row": (0, s), "column": (s, 0), "diagonal": (s / np.sqrt(2), s / np.sqrt(2))}[path]
        out.append(synthesize_view(lf, u0 + du, v0 + dv))
    return out


def dolly(lf: LightField, alpha: float, scale: float) -> np.ndarray:
    """Crude "move forward/backward" (slide 64): refocus at ``alpha`` then zoom the
    result by ``scale`` about the image centre (>1 = move forward / crop in).

    Approximation only: a true dolly changes parallax per depth (nearer things
    grow faster); a uniform zoom of the refocused image treats the scene as flat
    at the focal plane.
    """
    img = refocus(lf, alpha)
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), 0, scale)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(img.shape).astype(np.float32)
