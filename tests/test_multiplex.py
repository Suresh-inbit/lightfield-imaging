import numpy as np
import pytest
from lf import multiplex as mx
from lf.lightfield import LightField
from lf.synthetic import synthetic_light_field


@pytest.fixture(scope="module")
def lf3():
    return synthetic_light_field(n_views=3, size=64)


@pytest.fixture(scope="module")
def lf5():
    return synthetic_light_field(n_views=5, size=64)


@pytest.fixture(scope="module")
def lf7():
    base = synthetic_light_field(n_views=5, size=64)
    return LightField(base.views.reshape(1, 25, 64, 64, 3)[:, :7].copy(), "slice7")


@pytest.mark.parametrize("n", [3, 7, 11, 15, 23])
def test_s_matrix_properties(n):
    S = mx.s_matrix(n)
    assert set(np.unique(S)) <= {0.0, 1.0}
    assert np.all(S.sum(1) == (n + 1) // 2)
    assert np.allclose(S.T @ S, (n + 1) / 4 * (np.eye(n) + 1))


def test_s_matrix_invalid():
    with pytest.raises(ValueError):
        mx.s_matrix(9)


def _roundtrip(lf, W):
    Y = mx.capture(lf, W, photon_scale=None, read_noise=0.0)
    est = mx.demultiplex(Y, W, 1.0 / W.shape[0], (lf.nu, lf.nv))
    return np.abs(est.views - lf.views).max()


def test_noise_free_roundtrip(lf3, lf5, lf7):
    assert _roundtrip(lf3, mx.identity_patterns(9)) < 1e-5
    assert _roundtrip(lf3, mx.optimize_patterns(9, iters=2000, rng=0)[0]) < 1e-5
    assert _roundtrip(lf5, mx.optimize_patterns(25, iters=2000, rng=0)[0]) < 1e-4
    assert _roundtrip(lf7, mx.s_matrix(7)) < 1e-5


def _psnr(lf, W, seed=0, **kw):
    return mx.simulate_snr(lf, W, trials=2, rng=seed, photon_scale=None, **kw)["psnr"]


def test_read_noise_gain_s_matrix(lf7):
    pred = 20 * np.log10(mx.snr_gain(mx.s_matrix(7), "read"))     # ~3.6 dB
    got = _psnr(lf7, mx.s_matrix(7), read_noise=0.01) - _psnr(lf7, np.eye(7), read_noise=0.01)
    assert abs(got - pred) < 0.5


@pytest.mark.parametrize("fix,N", [("lf3", 9), ("lf5", 25)])
def test_read_noise_gain_optimised(request, fix, N):
    lf = request.getfixturevalue(fix)
    W = mx.optimize_patterns(N, iters=5000, rng=0)[0]
    pred = 20 * np.log10(mx.snr_gain(W, "read"))
    got = _psnr(lf, W, read_noise=0.01) - _psnr(lf, np.eye(N), read_noise=0.01)
    assert pred > 2.5
    assert abs(got - pred) < 0.5


def test_expected_mse_identity_matches_mc(lf3):
    N = 9
    W = np.eye(N)
    r = mx.simulate_snr(lf3, W, photon_scale=None, read_noise=0.01, trials=3, rng=1)["mse"]
    assert abs(r / mx.expected_mse(W, "read", sigma_r=0.01, exposure=1.0 / N) - 1) < 0.15
    xm = lf3.views.mean(axis=(2, 3, 4))
    r = mx.simulate_snr(lf3, W, photon_scale=500.0, read_noise=0.0, trials=3, rng=2)["mse"]
    e = mx.expected_mse(W, "photon", x_mean=xm.ravel(), exposure=1.0 / N, photon_scale=500.0)
    assert abs(r / e - 1) < 0.15


def test_expected_mse_s_matrix_matches_mc(lf7):
    W = mx.s_matrix(7)
    r = mx.simulate_snr(lf7, W, photon_scale=None, read_noise=0.01, trials=3, rng=3)["mse"]
    assert abs(r / mx.expected_mse(W, "read", sigma_r=0.01, exposure=1.0 / 7) - 1) < 0.15
    xm = lf7.views.mean(axis=(2, 3, 4)).ravel()
    r = mx.simulate_snr(lf7, W, photon_scale=500.0, read_noise=0.0, trials=3, rng=4)["mse"]
    e = mx.expected_mse(W, "photon", x_mean=xm, exposure=1.0 / 7, photon_scale=500.0)
    assert abs(r / e - 1) < 0.15


def test_closed_form_s_gain():
    for n in (7, 15):
        assert np.isclose(mx.snr_gain(mx.s_matrix(n), "read"), (n + 1) / (2 * np.sqrt(n)))
    assert mx.snr_gain(mx.s_matrix(15), "photon", x_mean=0.5) < 1.0   # photon noise: no gain


def test_theory_table_slide_ratios():
    t = mx.theory_table(15)
    r = t["read"]
    assert np.isclose(r["pa_nomux"]["snr_rel"], 1 / 15)
    assert np.isclose(r["pa_mux"]["snr_rel_binned"], (16 / (2 * np.sqrt(15))) / np.sqrt(15))
    assert np.isclose(t["photon"]["pa_nomux"]["snr_rel"], 1 / np.sqrt(15))
