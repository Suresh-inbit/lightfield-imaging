"""High-dynamic-range (HDR) imaging and HDR light fields.

Conventions
-----------
Library convention (docs/conventions.md) is ``views`` in [0, 1].  HDR light
fields are the one exception: their ``views`` hold *linear scene radiance*
(float32, unbounded, ``meta['radiance'] = True``).  LDR captures are in [0, 1]
after the camera response function (CRF).

Camera model (Debevec & Malik 1997)::

    Z = f( E * dt + noise ),  clipped at 1 (saturation), quantised

where E is scene radiance (irradiance on the sensor), dt the exposure time
(here "exposure" is any gain: shutter, ISO, or open aperture area) and f the
CRF.  Throughout, ``g = ln f^-1`` so ``g(Z) = ln E + ln dt``.

Sections: building blocks, simulation, synthetic HDR scene, light-field
strategies A-D, metrics.  See docs/09_hdr_light_field.md.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import shift as nd_shift, map_coordinates, gaussian_filter
from .lightfield import LightField
from .synthetic import _texture
from . import refocus as _rf

__all__ = [
    "weight_hat", "sample_pixels", "solve_response", "merge_hdr", "tonemap",
    "exposure_fusion", "simulate_capture", "inverse_crf", "synthetic_hdr_light_field",
    "hdr_from_ldr_light_field", "exposure_pattern", "capture_bracketed_views",
    "merge_bracketed_views", "merge_bracketed_refocused", "strategy_a", "hdr_refocus",
    "ldr_refocus_comparison", "temporal_bracket_hdr_lf", "aperture_area_bracketing",
    "log_rmse", "psnr_tonemapped", "evaluate_hdr_lf",
]


# ----------------------------------------------------------------------------
# Camera response helpers
# ----------------------------------------------------------------------------
def _srgb_fwd(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _srgb_inv(z):
    z = np.clip(z, 0, 1)
    return np.where(z <= 0.04045, z / 12.92, np.power((z + 0.055) / 1.055, 2.4))


def _crf_fwd(crf):
    if callable(crf):
        return crf
    if crf == "gamma":
        return lambda x: np.power(np.clip(x, 0, 1), 1 / 2.2)
    if crf == "srgb":
        return _srgb_fwd
    if crf == "linear":
        return lambda x: np.clip(x, 0, 1)
    raise ValueError(f"unknown crf {crf!r}")


def inverse_crf(crf="gamma"):
    """Return the inverse camera response ``z -> linear signal`` for a named CRF
    ('gamma' = z**2.2, 'srgb', 'linear'); a callable is assumed to be f already
    and is returned unchanged."""
    if callable(crf):
        return crf
    if crf == "gamma":
        return lambda z: np.power(np.clip(z, 0, 1), 2.2)
    if crf == "srgb":
        return _srgb_inv
    if crf == "linear":
        return lambda z: np.clip(z, 0, 1)
    raise ValueError(f"unknown crf {crf!r}")


def _ln_signal(z, response):
    """ln(E * dt) estimate for pixel values ``z`` in [0,1].

    ``response``: None (linear sensor), a name ('gamma', 'srgb', 'linear'), a
    callable inverse CRF ``z -> E*dt``, or a 1-D array ``g`` from
    :func:`solve_response` (``g[k] = ln(E dt)`` at level k)."""
    z = np.asarray(z, np.float64)
    if response is None:
        response = "linear"
    if isinstance(response, str) or callable(response):
        inv = inverse_crf(response)
        return np.log(np.maximum(inv(z), 1e-8))
    g = np.asarray(response, np.float64)
    return np.interp(z * (len(g) - 1), np.arange(len(g)), g)


# ----------------------------------------------------------------------------
# Debevec & Malik
# ----------------------------------------------------------------------------
def weight_hat(z, zmin: float = 0.0, zmax: float = 1.0):
    """Debevec-Malik hat weight, peak 1 at the mid-range.

    ``w(z) = (z - zmin)/(zmid - zmin)`` for ``z <= zmid`` and
    ``(zmax - z)/(zmax - zmid)`` above, ``zmid = (zmin+zmax)/2``; zero outside
    ``[zmin, zmax]``.  Down-weights under-exposed (noisy) and saturated
    (clipped) pixels.  Reference: Debevec & Malik, SIGGRAPH 1997, Eq. 4.
    """
    z = np.asarray(z, np.float64)
    zmid = 0.5 * (zmin + zmax)
    w = np.where(z <= zmid, (z - zmin) / (zmid - zmin), (zmax - z) / (zmax - zmid))
    return np.clip(w, 0.0, 1.0)


def sample_pixels(images, n: int = 200, rng=None, n_levels: int = 256, channel: int = 1):
    """Pick ``n`` pixel locations spread over the brightness range and return
    their quantised values in every image: shape ``(n, P)`` ints in
    ``[0, n_levels-1]``.  A pool of random pixels is sorted by mean value and
    evenly subsampled so that all brightness levels are represented.
    ``images`` : sequence of (H,W[,C]) arrays in [0,1]."""
    rng = np.random.default_rng(0) if rng is None else rng
    arr = []
    for im in images:
        im = np.asarray(im)
        if im.ndim == 3:
            im = im[..., min(channel, im.shape[-1] - 1)]
        arr.append(im)
    st = np.stack(arr, -1)                                  # (H,W,P)
    flat = st.reshape(-1, st.shape[-1])
    pool = rng.choice(len(flat), size=min(len(flat), 20 * n), replace=False)
    pool = pool[np.argsort(flat[pool].mean(1))]
    sel = pool[np.linspace(0, len(pool) - 1, n).astype(int)]
    return np.clip(np.round(flat[sel] * (n_levels - 1)), 0, n_levels - 1).astype(int)


def solve_response(samples, exposures, lam: float = 50.0, n_levels: int = 256):
    """Recover the log inverse response ``g`` by least squares.

    Minimise over ``g`` and ``ln E_i``::

        sum_ij w(Z_ij) [ g(Z_ij) - ln E_i - ln dt_j ]^2  +  lam * sum_z w(z) g''(z)^2

    with ``g(z_mid) = 0`` fixing the arbitrary scale; ``g''`` is the discrete
    second difference.  Weights use the integer-scale hat (peak (n_levels-1)/2)
    exactly as in the paper, so ``lam=50`` has the published meaning.

    Parameters
    ----------
    samples : (n_pixels, P) ints in [0, n_levels-1] (see :func:`sample_pixels`).
    exposures : (P,) exposure times.

    Returns ``g`` of shape (n_levels,) and ``lnE`` (n_pixels,).
    Reference: Debevec & Malik, "Recovering high dynamic range radiance maps
    from photographs", SIGGRAPH 1997, Sec. 2.1.
    """
    Z = np.asarray(samples, int)
    n, P = Z.shape
    lndt = np.log(np.asarray(exposures, np.float64))
    wint = lambda k: weight_hat(np.asarray(k, float) / (n_levels - 1)) * (n_levels - 1) / 2.0
    rows = n * P + (n_levels - 2) + 1
    A = np.zeros((rows, n_levels + n))
    b = np.zeros(rows)
    k = 0
    for i in range(n):
        for j in range(P):
            w = np.sqrt(wint(Z[i, j]))
            A[k, Z[i, j]] = w
            A[k, n_levels + i] = -w
            b[k] = w * lndt[j]
            k += 1
    A[k, n_levels // 2] = 1.0                                # g(mid) = 0
    k += 1
    for z in range(1, n_levels - 1):                         # smoothness
        w = np.sqrt(lam * wint(z))
        A[k, z - 1], A[k, z], A[k, z + 1] = w, -2 * w, w
        k += 1
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x[:n_levels], x[n_levels:]


def _as_stack(images):
    st = np.stack([np.asarray(im, np.float32) for im in images])
    if st.ndim == 3:
        st = st[..., None]
    return st


def _fallback_weight(z, t, tmin, tmax):
    """Tiny tie-breaking weight used only where the hat weight is ~0 everywhere:
    saturated -> prefer the shortest exposure, dark -> prefer the longest."""
    sat = z >= 0.5
    return 1e-6 * np.where(sat, tmin / t, t / tmax) ** 6


def _weights(z, kind, t, tmin, tmax):
    if callable(kind):
        w = kind(z)
    elif kind == "hat":
        w = weight_hat(z)
    elif kind == "uniform":
        w = ((z > 0.0) & (z < 1.0)).astype(np.float64)
    else:
        raise ValueError(f"unknown weights {kind!r}")
    return w + _fallback_weight(z, t, tmin, tmax)


def merge_hdr(images, exposures, response=None, weights="hat"):
    """Merge an exposure stack into a radiance map (H, W, C) float32.

    ``ln E = sum_j w(Z_j) [ g(Z_j) - ln dt_j ] / sum_j w(Z_j)`` (Debevec & Malik
    Eq. 6).  ``response``: None = linear sensor, a CRF name, callable inverse
    CRF or a ``g`` array.  Pixels where every exposure is saturated (or dark)
    get zero hat weight; they fall back to the shortest (longest) exposure
    estimate, a lower (upper) bound on the truth rather than NaN."""
    st = _as_stack(images)
    t = np.asarray(exposures, np.float64)
    num = np.zeros(st.shape[1:], np.float64)
    den = np.zeros(st.shape[1:], np.float64)
    for z, tj in zip(st, t):
        w = _weights(z.astype(np.float64), weights, tj, t.min(), t.max())
        num += w * (_ln_signal(z, response) - np.log(tj))
        den += w
    return np.exp(num / den).astype(np.float32)


def tonemap(radiance, method: str = "reinhard", key: float = 0.18, gamma: float = 2.2,
            white=None, lbar=None, eps: float = 1e-6):
    """Display image in [0,1] from linear radiance (H,W,C).

    * 'reinhard': global operator ``L = key/Lbar * Lw``, ``Ld = L (1 + L/white^2)/(1+L)``
      with ``Lbar = exp(mean ln(eps + Lw))`` the log-average luminance
      (Reinhard et al., SIGGRAPH 2002).  ``white`` = smallest luminance mapped to
      pure white (default: max of scaled luminance).
    * 'log': ``Ld = ln(1 + L) / ln(1 + white)``.
    * 'gamma': ``Ld = min(L, 1)``  (auto-exposure by key/Lbar, then clip).
    Colour ratios are preserved (the operator acts on luminance), then the
    result is gamma-encoded with exponent 1/gamma.  ``lbar`` can be passed to
    reuse a reference image's exposure (used for fair comparisons)."""
    R = np.maximum(np.asarray(radiance, np.float64), 0)
    lum = R.mean(-1, keepdims=True) if R.shape[-1] == 1 else \
        (R[..., :3] @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    if lbar is None:
        lbar = np.exp(np.mean(np.log(eps + lum)))
    L = key / lbar * lum
    wh = float(L.max()) if white is None else float(white) * key / lbar
    if method == "reinhard":
        Ld = L * (1 + L / max(wh ** 2, 1e-12)) / (1 + L)
    elif method == "log":
        Ld = np.log1p(L) / np.log1p(max(wh, 1e-6))
    elif method == "gamma":
        Ld = np.minimum(L, 1.0)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = R * (Ld / (lum + eps))
    return np.power(np.clip(out, 0, 1), 1 / gamma).astype(np.float32)


# ----------------------------------------------------------------------------
# Mertens exposure fusion
# ----------------------------------------------------------------------------
def _cv(fn, img, *a):
    out = fn(np.ascontiguousarray(img, np.float32), *a)
    return out.reshape(out.shape[:2] + (img.shape[-1],)) if img.ndim == 3 else out


def _gauss_pyr(img, n):
    p = [img]
    for _ in range(n - 1):
        p.append(_cv(cv2.pyrDown, p[-1]))
    return p


def _lap_pyr(img, n):
    g = _gauss_pyr(img, n)
    out = []
    for i in range(n - 1):
        up = _cv(cv2.pyrUp, g[i + 1], (g[i].shape[1], g[i].shape[0]))
        out.append(g[i] - up)
    return out + [g[-1]]


def exposure_fusion(images, w_contrast: float = 1, w_saturation: float = 1,
                    w_exposure: float = 1, levels=None):
    """Mertens-Kautz-Van Reeth exposure fusion of LDR images in [0,1] (no CRF).

    Per-pixel quality ``W = C^wc S^ws E^we`` with contrast C = |Laplacian| of the
    grey image, saturation S = std over channels, well-exposedness
    ``E = prod_c exp(-(I_c - 0.5)^2 / (2 * 0.2^2))``.  Weights are normalised
    over the stack, then images are blended in a Laplacian pyramid with the
    Gaussian pyramid of the weights to avoid seams.  Reference: Mertens, Kautz &
    Van Reeth, Pacific Graphics 2007."""
    st = _as_stack(images)
    N, H, W, C = st.shape
    Ws = []
    for im in st:
        gry = im.mean(-1)
        Cn = np.abs(cv2.Laplacian(gry, cv2.CV_32F))
        S = im.std(-1) if C > 1 else np.ones_like(gry)
        E = np.exp(-((im - 0.5) ** 2) / (2 * 0.2 ** 2)).prod(-1)
        Ws.append((Cn + 1e-12) ** w_contrast * (S + 1e-12) ** w_saturation * (E + 1e-12) ** w_exposure)
    Ws = np.stack(Ws)
    Ws = Ws / Ws.sum(0, keepdims=True)
    if levels is None:
        levels = max(2, int(np.floor(np.log2(min(H, W)))) - 2)
    fused = None
    for im, w in zip(st, Ws):
        lp = _lap_pyr(im, levels)
        gp = _gauss_pyr(w[..., None].astype(np.float32), levels)
        cur = [l * g.reshape(g.shape[:2] + (1,)) for l, g in zip(lp, gp)]
        fused = cur if fused is None else [a + b for a, b in zip(fused, cur)]
    out = fused[-1]
    for lvl in range(levels - 2, -1, -1):
        out = _cv(cv2.pyrUp, out, (fused[lvl].shape[1], fused[lvl].shape[0])) + fused[lvl]
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate_capture(radiance, exposure: float, crf="gamma", photon_scale: float = 2000.0,
                     read_noise: float = 0.005, bits: int = 8, rng=None):
    """Simulate one LDR photograph of a radiance image.

    ``x = E * exposure``;  photon noise ``Poisson(photon_scale * x) / photon_scale``;
    read noise ``N(0, read_noise^2)``; clip to [0, 1] (saturation, the full-well
    limit); apply the CRF ``f`` ('gamma' = x^(1/2.2), 'srgb', 'linear' or a
    callable); quantise to ``bits``.  Returns float32 in [0, 1]."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.maximum(np.asarray(radiance, np.float64) * exposure, 0)
    if photon_scale and photon_scale > 0:
        x = rng.poisson(x * photon_scale).astype(np.float64) / photon_scale
    if read_noise:
        x = x + rng.normal(0.0, read_noise, x.shape)
    z = _crf_fwd(crf)(np.clip(x, 0, 1))
    q = 2 ** bits - 1
    return (np.round(np.clip(z, 0, 1) * q) / q).astype(np.float32)


# ----------------------------------------------------------------------------
# Synthetic HDR light field
# ----------------------------------------------------------------------------
def synthetic_hdr_light_field(n_views: int = 7, size: int = 128, seed: int = 0,
                              gains=(0.03, 1.0, 40.0), name: str = "synthetic_hdr") -> LightField:
    """Layered scene like :func:`lf.synthetic.synthetic_light_field` but with a
    per-layer radiance gain: dark grass background (d=-1, x gains[0]), mid text
    plane (d=0, x gains[1]) and a very bright near gravel disc (d=+1.5, x gains[2]).
    Dynamic range is ~1e3-1e4 : 1 and each layer is only well captured by one
    exposure.

    EXCEPTION to the [0,1] convention: ``views`` hold LINEAR RADIANCE, float32,
    not clipped (``meta['radiance']=True``).  ``meta['gt_disparity']`` is the
    central-view disparity, ``meta['dynamic_range']`` = max/min radiance."""
    rng = np.random.default_rng(seed)
    layers = [
        {"disparity": -1.0, "texture": "grass", "mask": ("full",), "gain": gains[0]},
        {"disparity": 0.0, "texture": "text", "mask": ("rect", 0.10, 0.10, 0.55, 0.60), "gain": gains[1]},
        {"disparity": 1.5, "texture": "gravel", "mask": ("disc", 0.68, 0.66, 0.20), "gain": gains[2]},
    ]
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W] / float(size)
    pad = int(np.ceil(max(abs(l["disparity"]) for l in layers) * n_views)) + 2
    texs, masks = [], []
    for l in layers:
        t = _texture(l["texture"], H + 2 * pad, W + 2 * pad, rng) * np.float32(l["gain"])
        k = l["mask"][0]
        if k == "full":
            m = np.ones((H, W), np.float32)
        elif k == "rect":
            _, y0, x0, y1, x1 = l["mask"]
            m = ((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)).astype(np.float32)
        else:
            _, cy, cx, r = l["mask"]
            m = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)
        mp = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
        mp[pad:pad + H, pad:pad + W] = m
        if k == "full":                      # background covers the padded canvas
            mp[:] = 1.0
        texs.append(t); masks.append(mp)
    u0 = v0 = n_views // 2
    gt = np.zeros((H, W), np.float32)
    for l, m in zip(layers, masks):
        gt = np.where(m[pad:pad + H, pad:pad + W] > 0.5, l["disparity"], gt)
    views = np.zeros((n_views, n_views, H, W, 3), np.float32)
    for u in range(n_views):
        for v in range(n_views):
            img = np.zeros((H, W, 3), np.float32)
            for l, t, m in zip(layers, texs, masks):
                dy, dx = l["disparity"] * (u - u0), l["disparity"] * (v - v0)
                ts = nd_shift(t, (dy, dx, 0), order=1, mode="nearest")[pad:pad + H, pad:pad + W]
                ms = nd_shift(m, (dy, dx), order=1, mode="constant")[pad:pad + H, pad:pad + W][..., None]
                img = img * (1 - ms) + ts * ms
            views[u, v] = img
    meta = {"source": "synthetic HDR (lf.hdr)", "radiance": True, "gt_disparity": gt,
            "disparity_range": (-1.0, 1.5), "layers": layers,
            "dynamic_range": float(views.max() / max(views.min(), 1e-8))}
    return LightField(views, name, meta)


def hdr_from_ldr_light_field(lf: LightField, gamma: float = 2.2, boost_mask=None,
                             boost: float = 16.0, percentile: float = 98.0) -> LightField:
    """PSEUDO-radiance from a real LDR light field (demo only).

    Undo the display gamma (``E = z^gamma``) and brighten the brightest
    ``100-percentile`` % of the (d ~ 0) content of the central view by ``boost``
    (smoothly masked), imitating a specular highlight that the LDR camera
    clipped.  This is NOT real HDR: the information lost to clipping cannot be
    recovered; it only lets the refocusing experiments run on real data.
    ``boost_mask``: optional (H,W) weights in [0,1] overriding the automatic mask."""
    rad = np.power(np.clip(lf.views, 0, 1), gamma)
    c = lf.center_view
    lum = c.mean(-1)
    if boost_mask is None:
        region = np.ones_like(lum, bool)
        gt = lf.meta.get("gt_disparity")
        if gt is not None:
            region = np.abs(gt) < 0.5
        thr = np.percentile(lum[region], percentile)
        boost_mask = gaussian_filter(((lum >= thr) & region).astype(np.float32), 1.0)
    m = np.clip(np.asarray(boost_mask, np.float32), 0, 1)[None, None, :, :, None]
    rad = rad * (1 + (boost - 1) * m)
    meta = dict(lf.meta, radiance=True, pseudo_hdr=True)
    meta["dynamic_range"] = float(rad.max() / max(rad.min(), 1e-4))
    return LightField(rad.astype(np.float32), lf.name + "_pseudohdr", meta)


# ----------------------------------------------------------------------------
# Strategy A : one exposure per sub-aperture view
# ----------------------------------------------------------------------------
def exposure_pattern(nu: int, nv: int, kind: str = "checker",
                     exposures=(1 / 32, 1.0, 32.0), seed: int = 0):
    """(U, V) map of exposure times, one per sub-aperture view.

    'checker' : Latin-square diagonal pattern ``(u + 2v) mod k`` (every row and
        column of the angular grid sees all exposures, neighbours differ).
    'rows'    : horizontal bands of the angular grid.
    'radial'  : longest exposure at the centre, shortest at the rim -- mimics
        mechanical vignetting of a plenoptic camera in reverse (rim views get
        less light, i.e. act like short exposures).
    'random'  : balanced random assignment.
    ``exposures`` are sorted ascending.  Default (1/32, 1, 32) differs from the
    brief's (1/8, 1, 8) because the synthetic scene spans ~1e3:1 and 64:1
    cannot cover it."""
    ex = np.sort(np.asarray(exposures, float))
    k = len(ex)
    u, v = np.mgrid[0:nu, 0:nv]
    if kind == "checker":
        idx = (u + v) % 2 if k == 2 else (u + 2 * v) % k
    elif kind == "rows":
        idx = np.minimum(k - 1, u * k // nu)
    elif kind == "radial":
        r = np.hypot(u - nu // 2, v - nv // 2)
        idx = k - 1 - np.minimum(k - 1, (r / (r.max() + 1e-9) * k).astype(int))
    elif kind == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(nu * nv) % k).reshape(nu, nv)
    else:
        raise ValueError(f"unknown pattern {kind!r}")
    return ex[idx].astype(np.float64)


def capture_bracketed_views(radiance_lf: LightField, exposure_map, **sim):
    """Capture each sub-aperture view with its own exposure (programmable
    aperture / vignetted plenoptic camera).  Returns ``(ldr_lf, exposure_map)``;
    ``ldr_lf.meta['crf']`` records the CRF.  ``sim`` goes to
    :func:`simulate_capture` (``rng`` defaults to a seeded generator)."""
    rng = sim.pop("rng", None) or np.random.default_rng(0)
    crf = sim.get("crf", "gamma")
    em = np.asarray(exposure_map, float)
    out = np.empty_like(radiance_lf.views)
    for u in range(radiance_lf.nu):
        for v in range(radiance_lf.nv):
            out[u, v] = simulate_capture(radiance_lf.views[u, v], em[u, v], rng=rng, **sim)
    meta = dict(radiance_lf.meta, radiance=False, crf=crf, exposure_map=em)
    return LightField(out, radiance_lf.name + "_ldr", meta), em


def _target_disparity(d_c, du, dv, iters=3):
    """Disparity map as seen from a view offset (du, dv) from the centre.

    Forward-splat the centre map with a z-buffer (a point at (s,t) lands at
    ``(s + d du, t + d dv)``; the nearest, i.e. largest d, wins -- this resolves
    occlusions correctly).  Disoccluded holes are filled with the backward
    fixed point ``d_t(p) = d_c(p - d_t(p) (du, dv))`` (which cannot find
    foreground solutions by itself, hence the splat)."""
    if d_c is None:
        return None
    H, W = d_c.shape
    ss, tt = np.mgrid[0:H, 0:W].astype(np.float64)
    d0 = d_c.astype(np.float64)
    fp = d0.copy()
    for _ in range(iters):
        fp = map_coordinates(d0, [ss - fp * du, tt - fp * dv], order=1, mode="nearest")
    buf = np.full((H, W), -np.inf)
    ps, pt = ss + d0 * du, tt + d0 * dv
    for fs in (np.floor(ps), np.floor(ps) + 1):          # 4-neighbour splat, no cracks
        for ft in (np.floor(pt), np.floor(pt) + 1):
            ok = (fs >= 0) & (fs < H) & (ft >= 0) & (ft < W)
            np.maximum.at(buf, (fs[ok].astype(int), ft[ok].astype(int)), d0[ok])
    return np.where(np.isfinite(buf), buf, fp)


def merge_bracketed_views(ldr_lf: LightField, exposure_map, disparity=None, response=None,
                          robust: bool = True, sigma_ln: float = 0.5) -> LightField:
    """Merge a one-exposure-per-view capture into an HDR light field.

    For each target view ``(u, v)`` every source view ``(u', v')`` is warped
    onto it: ``Z'(s,t) = Z(s + d(s,t)(u'-u), t + d(s,t)(v'-v))`` by
    ``scipy.ndimage.map_coordinates`` (bilinear), where ``d`` is the disparity
    map (centre-view map re-projected to the target by a fixed-point iteration).
    ``ln(E dt)`` and hat weights are warped together (as ``w*ln`` and ``w``) and
    merged as in :func:`merge_hdr`.  If ``robust`` one extra pass down-weights
    sources whose ln-radiance differs from the first estimate by more than
    ``sigma_ln`` (Gaussian), which suppresses occlusion ghosts.

    ``disparity=None`` means "no depth": no warping at all (all views assumed
    aligned), which is wrong off the d=0 plane.  ``response=None`` uses
    ``ldr_lf.meta['crf']`` if present, else a linear sensor.
    Assumes a static scene and a known response."""
    em = np.asarray(exposure_map, float)
    if response is None:
        response = ldr_lf.meta.get("crf", "linear")
    U, V = ldr_lf.nu, ldr_lf.nv
    u0, v0 = ldr_lf.center
    H, W, C = ldr_lf.height, ldr_lf.width, ldr_lf.channels
    tmin, tmax = em.min(), em.max()
    lnE, wts = {}, {}
    for u in range(U):
        for v in range(V):
            z = ldr_lf.views[u, v].astype(np.float64)
            lnE[u, v] = _ln_signal(z, response) - np.log(em[u, v])
            wts[u, v] = weight_hat(z) + _fallback_weight(z, em[u, v], tmin, tmax)
    ss, tt = np.mgrid[0:H, 0:W].astype(np.float64)
    out = np.empty((U, V, H, W, C), np.float32)

    def combine(Ls, Ws, ref=None):
        Ws = np.stack(Ws); Ls = np.stack(Ls)
        if ref is not None:
            Ws = Ws * np.exp(-((Ls - ref) ** 2) / (2 * sigma_ln ** 2))
        return (Ws * Ls).sum(0) / np.maximum(Ws.sum(0), 1e-30)

    if disparity is None:                                    # no warp: same for every view
        est = combine(list(lnE.values()), list(wts.values()))
        if robust:
            est = combine(list(lnE.values()), list(wts.values()), est)
        out[:] = np.exp(est).astype(np.float32)[None, None]
        return LightField(out, ldr_lf.name + "_hdr", dict(ldr_lf.meta, radiance=True))
    d_c = np.asarray(disparity, np.float64)
    for u in range(U):
        for v in range(V):
            d = _target_disparity(d_c, u - u0, v - v0)
            Ls, Ws = [], []
            for up in range(U):
                for vp in range(V):
                    if (up, vp) == (u, v):
                        Ls.append(lnE[up, vp]); Ws.append(wts[up, vp]); continue
                    coords = [ss + d * (up - u), tt + d * (vp - v)]
                    wl = np.empty((H, W, C)); ww = np.empty((H, W, C))
                    for c in range(C):
                        wl[..., c] = map_coordinates(wts[up, vp][..., c] * lnE[up, vp][..., c],
                                                     coords, order=1, mode="nearest")
                        ww[..., c] = map_coordinates(wts[up, vp][..., c], coords, order=1, mode="nearest")
                    Ls.append(wl / np.maximum(ww, 1e-30)); Ws.append(ww)
            est = combine(Ls, Ws)
            if robust:
                est = combine(Ls, Ws, est)
            out[u, v] = np.exp(est)
    return LightField(out, ldr_lf.name + "_hdr", dict(ldr_lf.meta, radiance=True))


def merge_bracketed_refocused(ldr_lf: LightField, exposure_map, alpha: float, response=None):
    """Depth-free variant: merge in the refocused domain.  For every distinct
    exposure, shift views by ``-alpha (u-u0, v-v0)`` and average the linearised
    signal of that group (:func:`lf.refocus.refocus` with a group mask), giving
    one image per exposure; the images are then merged like an exposure stack
    (hat weights taken from the group's mean pixel value).  Exact only for the
    plane of disparity ``alpha``; elsewhere each group has a different blur
    pattern (different views), which produces inconsistent bokeh and ghosts.
    Returns the refocused radiance image (H, W, C)."""
    em = np.asarray(exposure_map, float)
    if response is None:
        response = ldr_lf.meta.get("crf", "linear")
    ln_lf = LightField(np.exp(_ln_signal(ldr_lf.views, response)).astype(np.float32))
    ts = np.unique(em)
    tmin, tmax = ts.min(), ts.max()
    num = den = 0
    for t in ts:
        m = (em == t).astype(np.float32)
        x = _rf.refocus(ln_lf, alpha, mask=m).astype(np.float64)
        zbar = _rf.refocus(ldr_lf, alpha, mask=m).astype(np.float64)
        w = weight_hat(zbar) + _fallback_weight(zbar, t, tmin, tmax)
        num = num + w * (np.log(np.maximum(x, 1e-8)) - np.log(t))
        den = den + w
    return np.exp(num / den).astype(np.float32)


def strategy_a(radiance_lf: LightField, kind: str = "checker", disparity="gt",
               exposures=(1 / 32, 1.0, 32.0), **sim) -> dict:
    """Strategy A end to end (1 shot).  ``disparity``: 'gt' (use
    ``meta['gt_disparity']``), None (no depth) or an (H,W) map."""
    d = radiance_lf.meta.get("gt_disparity") if isinstance(disparity, str) else disparity
    em = exposure_pattern(radiance_lf.nu, radiance_lf.nv, kind, exposures)
    ldr, em = capture_bracketed_views(radiance_lf, em, **sim)
    hdr = merge_bracketed_views(ldr, em, disparity=d)
    return dict(hdr_lf=hdr, ldr_lf=ldr, exposure_map=em, shots=1,
                log_rmse=evaluate_hdr_lf(hdr, radiance_lf)["mean"])


# ----------------------------------------------------------------------------
# Strategy B : refocusing and the linear-domain rule
# ----------------------------------------------------------------------------
def hdr_refocus(radiance_lf: LightField, alpha: float, tonemap_kwargs=None,
                return_radiance: bool = False):
    """Refocus in the LINEAR radiance domain, then tonemap:
    ``I = T( mean_{u,v} E(u,v, s+alpha(u-u0), t+alpha(v-v0)) )``.
    Averaging is a physical sum of photons, so it is only valid on linear
    values; ``T`` must come last.  Returns the display image (and the radiance
    image if ``return_radiance``)."""
    rad = _rf.refocus(radiance_lf, alpha)
    img = tonemap(rad, **(tonemap_kwargs or {}))
    return (img, rad) if return_radiance else img


def ldr_refocus_comparison(radiance_lf: LightField, alpha: float, exposure: float = 1.0,
                           hdr_lf: LightField | None = None, crf="gamma", **sim) -> dict:
    """Compare three orderings of (capture, tonemap, refocus).

    * ``ldr``  : each view captured once at ``exposure`` (clipped, gamma'd,
      quantised) and the *display* values averaged -- the naive JPEG pipeline.
    * ``ldr_lin`` : same capture, but linearised before averaging (fixes the
      gamma, cannot fix clipping).
    * ``hdr``  : refocus the radiance light field ``hdr_lf`` (default: the
      ground truth itself) in the linear domain.
    * ``tm_first`` : tonemap every view with a global operator, then refocus.
    Quantified by log-RMSE of the recovered radiance against the ground-truth
    refocused radiance; ``tm_first`` is scored by PSNR of the display image
    (its tone curve is not invertible without knowing the operator).
    Returns a dict of images, a difference map ``diff`` (|ldr - hdr display at
    the same exposure| in display units) and the scores."""
    gt = _rf.refocus(radiance_lf, alpha)
    sim = dict(sim)
    rng = sim.pop("rng", None) or np.random.default_rng(0)
    ldr_lf, _ = capture_bracketed_views(radiance_lf, np.full((radiance_lf.nu, radiance_lf.nv), exposure),
                                        crf=crf, rng=rng, **sim)
    ldr = _rf.refocus(ldr_lf, alpha)
    inv = inverse_crf(crf)
    ldr_lin = _rf.refocus(LightField(inv(ldr_lf.views)), alpha) / exposure
    ldr_rad = inv(ldr) / exposure
    ldr_lin_disp = _crf_fwd(crf)(np.clip(ldr_lin * exposure, 0, 1))
    hdr_rad = _rf.refocus(hdr_lf if hdr_lf is not None else radiance_lf, alpha)
    ref_disp = _crf_fwd(crf)(np.clip(gt * exposure, 0, 1))
    hdr_disp = _crf_fwd(crf)(np.clip(hdr_rad * exposure, 0, 1))
    # tonemap-first (parameters from the ground-truth centre view)
    c = radiance_lf.center_view
    lb = np.exp(np.mean(np.log(1e-6 + c.mean(-1))))
    tm_views = np.stack([[tonemap(radiance_lf.views[u, v], lbar=lb) for v in range(radiance_lf.nv)]
                         for u in range(radiance_lf.nu)])
    tm_first = _rf.refocus(LightField(tm_views), alpha)
    tm_ref = tonemap(gt, lbar=lb)
    psnr = lambda a, b: float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))
    return dict(gt=gt, ldr=ldr, ldr_lin=ldr_lin, ldr_radiance=ldr_rad, hdr=hdr_rad,
                hdr_display=hdr_disp, gt_display=ref_disp, tm_first=tm_first, tm_ref=tm_ref,
                diff=np.abs(ldr - hdr_disp).mean(-1),
                log_rmse_ldr=log_rmse(ldr_rad, gt), log_rmse_ldr_lin=log_rmse(ldr_lin, gt),
                log_rmse_hdr=log_rmse(hdr_rad, gt),
                psnr_display_ldr=psnr(ldr, ref_disp), psnr_display_ldr_lin=psnr(ldr_lin_disp, ref_disp),
                psnr_display_tm_first=psnr(tm_first, tm_ref))


# ----------------------------------------------------------------------------
# Strategy C : full temporal bracketing (upper bound)
# ----------------------------------------------------------------------------
def temporal_bracket_hdr_lf(radiance_lf: LightField, exposures=(1 / 32, 1.0, 32.0), **sim) -> dict:
    """Capture the whole light field at every exposure (len(exposures) shots)
    and merge per view with :func:`merge_hdr` (known CRF).  Static scene and
    perfectly repeatable views assumed; costs ``len(exposures)`` times the shots
    but needs no depth: the upper-bound reference for Strategy A."""
    sim = dict(sim)
    rng = sim.pop("rng", None) or np.random.default_rng(0)
    crf = sim.get("crf", "gamma")
    ldrs = []
    for t in exposures:
        l, _ = capture_bracketed_views(radiance_lf, np.full((radiance_lf.nu, radiance_lf.nv), t),
                                       rng=rng, **sim)
        ldrs.append(l)
    out = np.empty_like(radiance_lf.views)
    for u in range(radiance_lf.nu):
        for v in range(radiance_lf.nv):
            out[u, v] = merge_hdr([l.views[u, v] for l in ldrs], exposures, response=crf)
    hdr = LightField(out, radiance_lf.name + "_hdrC", dict(radiance_lf.meta, radiance=True))
    return dict(hdr_lf=hdr, ldr_lfs=ldrs, shots=len(exposures),
                log_rmse=evaluate_hdr_lf(hdr, radiance_lf)["mean"])


# ----------------------------------------------------------------------------
# Strategy D : aperture-area bracketing (programmable aperture)
# ----------------------------------------------------------------------------
def _open_cells(nu, nv, k):
    u0, v0 = nu // 2, nv // 2
    s = int(round(np.sqrt(k)))
    m = np.zeros((nu, nv), np.float32)
    if s * s == k and s <= min(nu, nv):
        m[u0 - s // 2:u0 - s // 2 + s, v0 - s // 2:v0 - s // 2 + s] = 1
    elif k <= nv:
        m[u0, v0 - k // 2:v0 - k // 2 + k] = 1
    else:
        raise ValueError(f"cannot open {k} cells on a {nu}x{nv} grid")
    return m


def aperture_area_bracketing(radiance_lf: LightField, areas=(1, 3, 9), alpha: float = 0.0,
                             shutter: float = 0.25, **sim) -> dict:
    """Programmable-aperture bracketing (Liang et al. 2008; cf. Nayar & Branzoi
    2003).  With ``k`` open cells the sensor collects ``sum`` of the ``k``
    sheared views, i.e. exposure ``= k * shutter`` of the mean view: images are
    the focal-plane photographs ``x_k = shutter * sum_{cells} L(...)``, captured
    with a fixed shutter time.  Merged like an exposure stack with exposure
    ``k * shutter``.  Open cells: 1 = centre, perfect squares = centred square
    block (9 = 3x3), others = a horizontal run.

    The result is HDR of the *refocused photo*, not of the light field, and it
    is only consistent in the in-focus plane because different ``k`` give
    different depth of field.  Its range is limited to ``max(areas)/min(areas)``
    (9:1 here, far below the scene's ~1e3:1).  Returns the merged radiance
    ``image`` (mean over the full open aperture ``max(areas)``), the captures,
    and the ground truth full-aperture refocus (scaled alike)."""
    sim = dict(sim)
    rng = sim.pop("rng", None) or np.random.default_rng(0)
    crf = sim.get("crf", "gamma")
    ldrs, ex = [], []
    for k in areas:
        m = _open_cells(radiance_lf.nu, radiance_lf.nv, k)
        mean_img = _rf.refocus(radiance_lf, alpha, mask=m)
        ex.append(k * shutter)
        ldrs.append(simulate_capture(mean_img, k * shutter, rng=rng, **sim))
    img = merge_hdr(ldrs, ex, response=crf)
    gt = _rf.refocus(radiance_lf, alpha)
    return dict(image=img, ldr=ldrs, exposures=np.array(ex), shots=len(areas), gt=gt,
                log_rmse=log_rmse(img, gt))


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def log_rmse(est, gt, eps: float = 1e-4) -> float:
    """``sqrt(mean( (ln(est+eps) - ln(gt+eps))^2 ))`` (natural log): relative
    error that weighs dark and bright regions equally; 0.69 = factor-2 error."""
    d = np.log(np.asarray(est, np.float64) + eps) - np.log(np.asarray(gt, np.float64) + eps)
    return float(np.sqrt(np.mean(d ** 2)))


def psnr_tonemapped(est, gt, **tm) -> float:
    """PSNR (peak 1) between tonemapped ``est`` and ``gt``; both use the
    log-average luminance of ``gt`` so the same exposure is applied."""
    gt = np.asarray(gt, np.float64)
    lum = gt.mean(-1)
    lb = np.exp(np.mean(np.log(1e-6 + lum)))
    a, b = tonemap(est, lbar=lb, **tm), tonemap(gt, lbar=lb, **tm)
    m = float(np.mean((a.astype(np.float64) - b) ** 2))
    return float("inf") if m == 0 else float(10 * np.log10(1.0 / m))


def evaluate_hdr_lf(est_lf: LightField, gt_lf: LightField, eps: float = 1e-4) -> dict:
    """Per-view and mean log-RMSE of an HDR light field against ground truth."""
    pv = np.array([[log_rmse(est_lf.views[u, v], gt_lf.views[u, v], eps)
                    for v in range(gt_lf.nv)] for u in range(gt_lf.nu)])
    return dict(per_view=pv, mean=float(pv.mean()), center=float(pv[gt_lf.center]))
