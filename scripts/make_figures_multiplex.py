"""Figures for docs/08_programmable_aperture.md (headless)."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lf
from lf import multiplex as mx
from lf.synthetic import synthetic_light_field
from lf.lightfield import LightField
from lf.refocus import refocus
from lf.utils import psnr

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
READ, PS = 0.01, 1000.0          # read-noise std, photons per unit signal


def fig_patterns():
    Wi = mx.identity_patterns(9)
    Wo, f = mx.optimize_patterns(9, iters=10000, rng=0)
    fig, axes = plt.subplots(2, 9, figsize=(14, 3.6))
    for r, (W, t) in enumerate([(Wi, "no multiplexing (1 cell / shot)"),
                                (Wo, "optimised (about half open)")]):
        for k in range(9):
            ax = axes[r, k]
            ax.imshow(mx.pattern_to_aperture(W[k], (3, 3)), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"shot {k + 1}", fontsize=8)
        axes[r, 0].set_ylabel(t, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle(f"3x3 aperture patterns (white = open); read-noise SNR gain of bottom row = "
                 f"{mx.snr_gain(Wo, 'read'):.2f}", fontsize=10)
    plt.tight_layout(); fig.savefig(OUT / "aperture_patterns.png", dpi=110); plt.close(fig)


def pipeline(lfi, N_side, fname, title):
    N = N_side ** 2
    gt = lfi.views
    Wi = np.eye(N)
    Wo, _ = mx.optimize_patterns(N, iters=10000, rng=0)
    rng = np.random.default_rng(0)
    Yi = mx.capture(lfi, Wi, 1.0, PS, READ, rng)
    Yo = mx.capture(lfi, Wo, 1.0, PS, READ, rng)
    ei = mx.demultiplex(Yi, Wi, 1.0 / N, (N_side, N_side))
    eo = mx.demultiplex(Yo, Wo, 1.0 / N, (N_side, N_side))
    c = lfi.center; j = c[0] * N_side + c[1]
    gtc = gt[c]
    # display the shots normalised by their number of open cells (mean aperture image)
    shot_i = Yi[j] * N / Wi[j].sum()
    k = int(np.argmax(Wo.sum(1)))
    shot_o = Yo[k] * N / Wo[k].sum()
    allp = lambda e: np.mean([psnr(e.views[u, v], gt[u, v]) for u in range(N_side) for v in range(N_side)])
    panels = [(shot_i, "single-cell shot (noisy)\nPSNR %.1f" % psnr(np.clip(shot_i, 0, 1), gtc)),
              (ei.views[c], "sequential, demultiplexed\nPSNR %.1f (all views %.1f)" % (psnr(np.clip(ei.views[c], 0, 1), gtc), allp(ei))),
              (shot_o, "multiplexed shot (%d cells open)\nblurred, less noisy" % Wo[k].sum()),
              (eo.views[c], "multiplexed, demultiplexed\nPSNR %.1f (all views %.1f)" % (psnr(np.clip(eo.views[c], 0, 1), gtc), allp(eo))),
              (gtc, "ground-truth view")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1)); ax.set_title(t, fontsize=9); ax.axis("off")
    fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout(); fig.savefig(OUT / fname, dpi=110); plt.close(fig)
    print(fname, "all-view PSNR identity %.2f / mux %.2f" % (allp(ei), allp(eo)))


def fig_snr_vs_N():
    base = synthetic_light_field(n_views=5, size=48)
    Ns = [3, 7, 9, 11, 15, 19, 23, 25]
    regimes = {"read noise only": dict(noise="read", sigma_r=READ, ps=None, ro=READ),
               "photon noise only": dict(noise="photon", sigma_r=0.0, ps=PS, ro=0.0),
               "mixed": dict(noise="mixed", sigma_r=READ, ps=PS, ro=READ)}
    xm = float(base.views.mean())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    rows = []
    for ax, (rname, rc) in zip(axes, regimes.items()):
        th_s, th_o, th_p, sim_s, sim_o, sim_p = [], [], [], [], [], []
        for N in Ns:
            sub = LightField(base.views.reshape(1, 25, 48, 48, 3)[:, :N].copy())
            xmv = sub.views.mean(axis=(2, 3, 4)).ravel()
            kw = dict(sigma_r=rc["sigma_r"], x_mean=xmv, photon_scale=PS if rc["ps"] else 1e12)
            Wh = mx.hadamard_patterns(N)
            Wo = mx.optimize_patterns(N, iters=8000, rng=0, restarts=4)[0]
            is_s = N in mx.valid_s_sizes(64)
            # theory
            m_id = mx.expected_mse(np.eye(N), rc["noise"], exposure=1.0 / N, **kw)
            m_pl = mx.expected_mse(np.eye(N), rc["noise"], exposure=1.0, **kw)
            g = lambda W: np.sqrt(m_id / mx.expected_mse(W, rc["noise"], exposure=1.0 / N, **kw))
            th_s.append(g(Wh) if is_s else np.nan); th_o.append(g(Wo)); th_p.append(np.sqrt(m_id / m_pl))
            # simulation
            sk = dict(photon_scale=rc["ps"], read_noise=rc["ro"], trials=2, rng=N)
            mi = mx.simulate_snr(sub, np.eye(N), **sk)["mse"]
            sim_s.append(np.sqrt(mi / mx.simulate_snr(sub, Wh, **sk)["mse"]) if is_s else np.nan)
            sim_o.append(np.sqrt(mi / mx.simulate_snr(sub, Wo, **sk)["mse"]))
            sim_p.append(np.sqrt(mi / mx.simulate_snr(sub, None, plenoptic=True, **sk)["mse"]))
            rows.append((rname, N, th_s[-1], sim_s[-1], th_o[-1], sim_o[-1], th_p[-1], sim_p[-1]))
        ax.axhline(1, color="k", lw=1, label="PA, no multiplexing (=1)")
        ax.plot(Ns, th_p, "-", color="tab:green", label="plenoptic (theory)")
        ax.plot(Ns, sim_p, "x", color="tab:green")
        ax.plot(Ns, th_s, "-", color="tab:blue", label="S-matrix (theory)")
        ax.plot(Ns, sim_s, "o", color="tab:blue", mfc="none", label="S-matrix (sim)")
        ax.plot(Ns, th_o, "--", color="tab:red", label="optimised (theory)")
        ax.plot(Ns, sim_o, "s", color="tab:red", mfc="none", label="optimised (sim)")
        ax.set_yscale("log"); ax.set_xlabel("N (aperture cells = shots)"); ax.set_title(rname)
        ax.set_ylabel("SNR gain vs. sequential capture"); ax.grid(alpha=.3, which="both")
    axes[0].legend(fontsize=7)
    plt.tight_layout(); fig.savefig(OUT / "multiplex_snr_vs_N.png", dpi=110); plt.close(fig)
    print("regime N thS simS thO simO thPlen simPlen")
    for r in rows:
        if r[1] in (7, 9, 15, 25):
            print(r[0], r[1], *["%.3f" % v for v in r[2:]])


def fig_refocus():
    lfi = synthetic_light_field(n_views=5, size=96)
    N = 25
    Wi = np.eye(N); Wo, _ = mx.optimize_patterns(N, iters=10000, rng=0)
    rng = np.random.default_rng(1)
    ei = mx.demultiplex(mx.capture(lfi, Wi, 1.0, PS, READ, rng), Wi, 1 / N, (5, 5))
    eo = mx.demultiplex(mx.capture(lfi, Wo, 1.0, PS, READ, rng), Wo, 1 / N, (5, 5))
    a = 1.5
    ref = refocus(lfi, a)
    ri, ro = refocus(ei, a), refocus(eo, a)
    c = lfi.center
    ps = lambda e: np.mean([psnr(e.views[u, v], lfi.views[u, v]) for u in range(5) for v in range(5)])
    panels = [(ei.views[c], "sequential: one view\nPSNR %.1f" % psnr(ei.views[c], lfi.views[c])),
              (eo.views[c], "multiplexed: one view\nPSNR %.1f" % psnr(eo.views[c], lfi.views[c])),
              (ri, "sequential, refocused\nPSNR %.1f" % psnr(ri, ref)),
              (ro, "multiplexed, refocused\nPSNR %.1f" % psnr(ro, ref)),
              (ref, "clean LF, refocused (alpha=1.5)")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1)); ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout(); fig.savefig(OUT / "multiplex_refocus.png", dpi=110); plt.close(fig)
    print("refocus PSNR seq %.2f mux %.2f ; per-view seq %.2f mux %.2f" %
          (psnr(ri, ref), psnr(ro, ref), ps(ei), ps(eo)))


if __name__ == "__main__":
    t0 = time.time()
    fig_patterns()
    syn = synthetic_light_field(n_views=3, size=160)
    pipeline(syn, 3, "multiplex_pipeline.png",
             "Synthetic 3x3 LF, T=1, read noise %.3f, %g photons/unit (N=9 shots)" % (READ, PS))
    fig_snr_vs_N()
    fig_refocus()
    if lf.io.available().get("lego"):
        real = lf.io.load("lego", downsample=4, angular_crop=5)
        pipeline(real, 5, "multiplex_pipeline_lego.png",
                 "Lego Knights 5x5 LF (Stanford), N=25 shots, same noise model")
    else:
        print("lego not available; real-data figure skipped (synthetic only)")
    print("done in %.1fs" % (time.time() - t0))
