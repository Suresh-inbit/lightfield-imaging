"""Programmable-aperture photography and multiplexed light-field acquisition.

Liang, Lin, Wong, Liu, Chen, "Programmable Aperture Photography: Multiplexed
Light Field Acquisition", SIGGRAPH 2008 (EE 5176 lecture 16, slides 21-37).

Model (see docs/08_programmable_aperture.md)
-------------------------------------------
The aperture is divided into N cells (u, v).  Cell j has radiance image
``x_j = lf.views[u, v]`` (per unit time).  We take K shots of duration
``tau = T / K`` (same total exposure T for every scheme).  In shot k the cells
with ``W[k, j] == 1`` are open:

    s_k = tau * sum_j W[k, j] x_j                 (noise-free signal)
    y_k = Poisson(photon_scale * s_k)/photon_scale + N(0, sigma_r^2)
    x_hat = pinv(W) y / tau

``W = I`` is plain sequential capture (slides 21-24); an S-matrix (about half
the cells open) is the multiplexed capture of slides 25-28.  Everything is
per pixel, so spatial resolution is the full sensor resolution.

Noise conventions (used by all closed forms below)
-------------------------------------------------
Plenoptic baseline == identity pattern with per-sample exposure T (K=1 time
slice, N pixels).  Sequential / multiplexed capture uses tau = T / K.
"""
from __future__ import annotations
import numpy as np
from .lightfield import LightField

__all__ = ["s_matrix", "hadamard_patterns", "identity_patterns", "random_patterns",
           "optimize_patterns", "pattern_to_aperture", "capture", "capture_plenoptic",
           "demultiplex", "expected_mse", "snr_gain", "theory_table", "simulate_snr",
           "resolution_penalty", "valid_s_sizes"]


# ----------------------------------------------------------------------------
# Pattern construction
# ----------------------------------------------------------------------------
def _is_prime(q: int) -> bool:
    return q > 1 and all(q % p for p in range(2, int(q ** 0.5) + 1))


def valid_s_sizes(nmax: int = 64) -> list[int]:
    """Sizes n <= nmax for which :func:`s_matrix` exists (2^k-1 or prime = 3 mod 4)."""
    out = []
    for n in range(3, nmax + 1):
        if (n + 1) & n == 0 or (_is_prime(n) and n % 4 == 3):
            out.append(n)
    return out


def _sylvester(m: int) -> np.ndarray:
    H = np.array([[1]])
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    return H


def s_matrix(n: int) -> np.ndarray:
    """Harwit-Sloane S-matrix of order ``n`` as a 0/1 float matrix.

    ``n = 2^k - 1``: drop the first row/column of the Sylvester Hadamard matrix
    of order n+1 and map -1 -> 1, +1 -> 0.
    ``n`` prime, ``n = 3 mod 4``: circulant whose first row is 1 at offset 0 and
    at the quadratic non-residues (Paley / difference-set construction).
    Both give ``(n+1)/2`` ones per row and ``S^T S = (n+1)/4 (I + J)``, hence
    ``tr((S^T S)^-1) = 4 n^2 / (n+1)^2``.

    Reference: Harwit & Sloane, Hadamard Transform Optics, 1979.
    """
    if n >= 3 and (n + 1) & n == 0:
        H = _sylvester(n + 1)
        return ((1 - H[1:, 1:]) // 2).astype(np.float64)
    if _is_prime(n) and n % 4 == 3:
        qr = {(i * i) % n for i in range(1, n)}
        row = np.array([1.0 if (d == 0 or d not in qr) else 0.0 for d in range(n)])
        return np.stack([np.roll(row, i) for i in range(n)])
    raise ValueError(f"no S-matrix construction for n={n}; valid: {valid_s_sizes(64)}")


def identity_patterns(n: int) -> np.ndarray:
    """No multiplexing: one open cell per shot (slides 21-24)."""
    return np.eye(n)


def random_patterns(n: int, density: float = 0.5, rng=None, max_tries: int = 1000) -> np.ndarray:
    """Random invertible 0/1 patterns with the given fraction of open cells."""
    rng = np.random.default_rng(rng)
    for _ in range(max_tries):
        W = (rng.random((n, n)) < density).astype(np.float64)
        if np.linalg.svd(W, compute_uv=False).min() > 1e-6:
            return W
    raise RuntimeError("could not draw an invertible random pattern matrix")


def _objective(W, noise, x_mean, sigma_r=1.0, photon_scale=1.0):
    s = np.linalg.svd(W, compute_uv=False)
    if s.min() < 1e-6:
        return np.inf
    f = 0.0
    if noise in ("read", "mixed"):
        f += sigma_r ** 2 * np.sum(1.0 / s ** 2)
    if noise in ("photon", "mixed"):
        P = np.linalg.pinv(W)
        d = W @ np.broadcast_to(np.asarray(x_mean, float), (W.shape[1],))
        f += np.sum((P ** 2) * d[None, :]) / photon_scale
    return float(f)


def optimize_patterns(n: int, iters: int = 10000, noise: str = "read", rng=None, init=None,
                      x_mean: float = 1.0, restarts: int = 1):
    """Simulated-annealing bit-flip search for an n x n 0/1 matrix minimising the MSE.

    Objective (up to the constant 1/(n tau^2) factor):
      read   : tr((W^T W)^-1)
      photon : tr((W^T W)^-1 W^T diag(W x_mean) W (W^T W)^-1)  [= tr(P diag(W x) P^T), P = W^+]
    Works for any n (e.g. 9 for a 3x3, 25 for a 5x5 aperture) where no S-matrix
    exists.  Start: ``init`` or a random half-open invertible matrix (for read
    noise a half-open start is near the optimum; for photon noise pass
    ``init=np.eye(n)`` or let the search move toward sparse rows).

    ``restarts`` runs independent searches and keeps the best.  Caveat: the
    landscape is rugged; for n where an S-matrix exists (7, 15, ...) the
    annealer typically stays 10-25 % short of its SNR gain, so prefer
    :func:`s_matrix` there.

    Returns ``(W, objective)`` for the best W seen.
    Reference: Schechner, Nayar, Belhumeur, PAMI 2007; Ratner & Schechner 2007.
    """
    rng = np.random.default_rng(rng)
    if restarts > 1:
        runs = [optimize_patterns(n, iters, noise, rng, init, x_mean) for _ in range(restarts)]
        return min(runs, key=lambda r: r[1])
    W = random_patterns(n, 0.5, rng) if init is None else np.array(init, float)
    f = _objective(W, noise, x_mean)
    if not np.isfinite(f):
        raise ValueError("init is singular")
    best, fbest = W.copy(), f
    T0 = 0.002 * f
    T1 = 1e-5 * f
    for it in range(iters):
        temp = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        i, j = rng.integers(n), rng.integers(n)
        W[i, j] = 1.0 - W[i, j]
        g = _objective(W, noise, x_mean)
        if g < f or rng.random() < np.exp(-(g - f) / temp):
            f = g
            if f < fbest:
                best, fbest = W.copy(), f
        else:
            W[i, j] = 1.0 - W[i, j]
    return best, fbest


def hadamard_patterns(n_views: int) -> np.ndarray:
    """S-matrix patterns for ``n_views`` aperture cells.

    Uses :func:`s_matrix` when it exists (n = 2^k-1 or prime = 3 mod 4, e.g. 3, 7,
    11, 15, 19, 23, 31).  Otherwise (e.g. 9 or 25) falls back to the annealed
    read-noise optimum (:func:`optimize_patterns`, deterministic seed).
    """
    try:
        return s_matrix(n_views)
    except ValueError:
        return optimize_patterns(n_views, iters=10000, noise="read", rng=0)[0]


def pattern_to_aperture(w_row, uv) -> np.ndarray:
    """Reshape one pattern row to a (U, V) aperture image (1 = open)."""
    return np.asarray(w_row).reshape(uv)


# ----------------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------------
def _noisy(s, photon_scale, read_noise, rng):
    out = s
    if photon_scale is not None and np.isfinite(photon_scale):
        out = rng.poisson(np.maximum(photon_scale * s, 0)).astype(np.float64) / photon_scale
    if read_noise:
        out = out + rng.normal(0.0, read_noise, size=s.shape)
    return out


def capture(lf: LightField, W, total_exposure: float = 1.0, photon_scale: float = 1000.0,
            read_noise: float = 0.01, rng=None, gain: float = 1.0) -> np.ndarray:
    """Simulate K programmable-aperture shots; returns (K, H, W, C) float32.

    ``s_k = (T/K) * gain * sum_j W[k,j] x_j``; then Poisson noise with
    ``photon_scale`` photons per unit signal (``None``/inf disables it) and
    Gaussian read noise of std ``read_noise`` (same units as the signal).
    Cell j <-> ``lf.views[u, v]`` with ``j = u * V + v``.

    Reference: Liang et al., SIGGRAPH 2008, Sec. 3.
    """
    rng = np.random.default_rng(rng)
    W = np.asarray(W, np.float64)
    K, N = W.shape
    if N != lf.nu * lf.nv:
        raise ValueError(f"W has {N} columns but the LF has {lf.nu * lf.nv} views")
    tau = total_exposure / K
    x = lf.views.reshape(N, -1).astype(np.float64)
    s = gain * tau * (W @ x)
    y = _noisy(s, photon_scale, read_noise, rng)
    return y.reshape((K,) + lf.views.shape[2:]).astype(np.float32)


def capture_plenoptic(lf: LightField, total_exposure: float = 1.0, photon_scale: float = 1000.0,
                      read_noise: float = 0.01, rng=None) -> LightField:
    """Baseline: all N angular samples recorded at once, each for the full time T.

    Signal per sample is ``T x_j`` with the same Poisson + read noise model, with
    no T/K penalty.  The returned LightField is the estimate ``y / T``.
    NOTE: this ignores the plenoptic camera's N-fold loss of spatial resolution
    (M/N pixels per view); :func:`resolution_penalty` quantifies that instead.
    """
    rng = np.random.default_rng(rng)
    s = total_exposure * lf.views.astype(np.float64)
    y = _noisy(s, photon_scale, read_noise, rng) / total_exposure
    return LightField(y.astype(np.float32), lf.name + "_plenoptic", dict(lf.meta))


def resolution_penalty(M: float, N: int) -> dict:
    """Spatial resolution of one view: plenoptic M/N pixels, programmable aperture M.

    E.g. Lytro Illum 16 MP, N = 14 x 14 -> about 0.08 MP per view (slide 6/14).
    """
    return {"plenoptic_pixels": M / N, "programmable_pixels": M, "ratio": float(N)}


# ----------------------------------------------------------------------------
# Inversion
# ----------------------------------------------------------------------------
def demultiplex(Y, W, exposure_per_shot: float, angular=None, ridge: float = 0.0,
                clip: bool = False, name: str = "demultiplexed") -> LightField:
    """Least-squares inversion ``x_hat = (W^T W + ridge I)^-1 W^T y / tau``.

    Y has shape (K, H, W, C); applied to all pixels at once by reshaping to
    (K, -1).  ``angular=(U, V)`` (default: square root of N).  ``ridge=0`` uses
    the pseudo-inverse.  ``clip`` clamps to [0, 1].  Output is a LightField.
    """
    Y = np.asarray(Y)
    W = np.asarray(W, np.float64)
    K, N = W.shape
    if Y.shape[0] != K:
        raise ValueError("Y first axis must equal number of patterns")
    if angular is None:
        r = int(round(np.sqrt(N)))
        if r * r != N:
            raise ValueError("pass angular=(U, V) for non-square N")
        angular = (r, r)
    if ridge > 0:
        P = np.linalg.solve(W.T @ W + ridge * np.eye(N), W.T)
    else:
        P = np.linalg.pinv(W)
    X = (P @ Y.reshape(K, -1).astype(np.float64)) / exposure_per_shot
    if clip:
        X = np.clip(X, 0, 1)
    views = X.reshape(tuple(angular) + Y.shape[1:]).astype(np.float32)
    return LightField(views, name)


# ----------------------------------------------------------------------------
# Closed-form analysis
# ----------------------------------------------------------------------------
def expected_mse(W, noise: str = "read", sigma_r: float = 0.01, x_mean=0.5,
                 exposure: float = 1.0, photon_scale: float = 1000.0) -> float:
    """Expected per-sample MSE of ``x_hat = W^+ y / tau`` (average over the N cells).

    ``exposure`` is the PER-SHOT time ``tau`` (use T/K for K shots; for the
    plenoptic baseline use W = I with exposure = T).  With P = W^+:
      read   : sigma_r^2 / tau^2 * tr(P P^T) / N
      photon : 1/(tau * photon_scale) * tr(P diag(W x_mean) P^T) / N
      mixed  : sum of both.
    ``x_mean`` is a scalar or length-N vector of mean cell radiances (photon
    variance is linear in x, so averaging over pixels is exact).
    """
    W = np.asarray(W, np.float64)
    K, N = W.shape
    P = np.linalg.pinv(W)
    xm = np.broadcast_to(np.asarray(x_mean, float), (N,))
    out = 0.0
    if noise in ("read", "mixed"):
        out += sigma_r ** 2 / exposure ** 2 * np.sum(P ** 2) / N
    if noise in ("photon", "mixed"):
        out += np.sum((P ** 2) * (W @ xm)[None, :]) / (exposure * photon_scale) / N
    return float(out)


def snr_gain(W, noise: str = "read", sigma_r: float = 0.01, x_mean=0.5,
             photon_scale: float = 1000.0, total_exposure: float = 1.0) -> float:
    """SNR (amplitude) gain of W over identity patterns at the same total exposure.

    ``sqrt(MSE_identity / MSE_W)`` with tau = T/N for the identity and T/K for W.
    For an S-matrix and read noise this is ``(N+1) / (2 sqrt N)``.
    """
    W = np.asarray(W, np.float64)
    K, N = W.shape
    kw = dict(noise=noise, sigma_r=sigma_r, x_mean=x_mean, photon_scale=photon_scale)
    m_id = expected_mse(np.eye(N), exposure=total_exposure / N, **kw)
    m_w = expected_mse(W, exposure=total_exposure / K, **kw)
    return float(np.sqrt(m_id / m_w))


def theory_table(N: int, sigma_r: float = 0.01, photon_scale: float = 1000.0,
                 x_mean: float = 0.5, total_exposure: float = 1.0, W=None) -> dict:
    """Slide-29 style comparison from closed forms.

    For each regime (photon / read / mixed) returns, for plenoptic, PA without
    multiplexing and PA with multiplexing (W = S-matrix or optimised):
    ``mse`` and the SNR relative to the plenoptic sample,
    ``snr_rel`` (per pixel, same pixel count as the plenoptic *sample*) and
    ``snr_rel_binned`` (= snr_rel * sqrt(N): the PA's N-fold more pixels are
    binned N:1 so both outputs have M/N pixels per view).
    The slide's claimed ratios are in ``slide`` (photon: 1, 1/sqrt N, 1/sqrt 2;
    read: 1, 1/N, 1/2), to be compared with the binned/unbinned columns.
    """
    Wm = hadamard_patterns(N) if W is None else np.asarray(W, float)
    T = total_exposure
    regimes = {"photon": dict(noise="photon", sigma_r=0.0),
               "read": dict(noise="read", sigma_r=sigma_r),
               "mixed": dict(noise="mixed", sigma_r=sigma_r)}
    out = {"N": N, "slide": {"photon": (1.0, 1 / np.sqrt(N), 1 / np.sqrt(2)),
                             "read": (1.0, 1.0 / N, 0.5)}}
    for rname, kw in regimes.items():
        kw = dict(kw, x_mean=x_mean, photon_scale=photon_scale)
        m_p = expected_mse(np.eye(N), exposure=T, **kw)
        m_n = expected_mse(np.eye(N), exposure=T / N, **kw)
        m_m = expected_mse(Wm, exposure=T / Wm.shape[0], **kw)
        row = {}
        for k, m in (("plenoptic", m_p), ("pa_nomux", m_n), ("pa_mux", m_m)):
            rel = float(np.sqrt(m_p / m))
            row[k] = dict(mse=m, snr_rel=rel, snr_rel_binned=rel * np.sqrt(N) if k != "plenoptic" else 1.0)
        out[rname] = row
    return out


def simulate_snr(lf: LightField, W, total_exposure: float = 1.0, photon_scale: float = 1000.0,
                 read_noise: float = 0.01, trials: int = 3, rng=None, plenoptic: bool = False,
                 ridge: float = 0.0) -> dict:
    """Monte-Carlo: capture, demultiplex and compare with ``lf.views``.

    Returns dict(mse, psnr, snr_db) where snr_db = 10 log10(mean(x^2)/mse),
    averaged over ``trials`` independent noise draws.  ``plenoptic=True`` uses
    :func:`capture_plenoptic` (W ignored).
    """
    rng = np.random.default_rng(rng)
    x = lf.views.astype(np.float64)
    mses = []
    for _ in range(trials):
        if plenoptic:
            est = capture_plenoptic(lf, total_exposure, photon_scale, read_noise, rng).views
        else:
            W_ = np.asarray(W, float)
            Y = capture(lf, W_, total_exposure, photon_scale, read_noise, rng)
            est = demultiplex(Y, W_, total_exposure / W_.shape[0], (lf.nu, lf.nv), ridge).views
        mses.append(np.mean((est.astype(np.float64) - x) ** 2))
    mse = float(np.mean(mses))
    return {"mse": mse, "psnr": float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
            "snr_db": float(10 * np.log10(np.mean(x ** 2) / mse)) if mse > 0 else float("inf")}
