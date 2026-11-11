"""Figures for docs/06_extended_dof.md and docs/07_depth_estimation.md (headless)."""
import pathlib, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lf
from lf import depth, edof, refocus
from lf.utils import gray

OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
SYN_ALPHAS = np.linspace(-1.5, 2.0, 15)
REAL_ALPHAS = {"lego": np.linspace(-3, 3, 25), "bracelet": np.linspace(-3, 3, 25), "lytro": np.linspace(-1.5, 1.5, 21)}
REAL_KW = {"lego": dict(downsample=4, angular_crop=9), "bracelet": dict(downsample=4, angular_crop=9),
           "lytro": dict(crop_angular=9, downsample=2)}


def timed(label, f):
    t = time.time(); r = f(); print(f"  {label}: {time.time() - t:.1f}s"); return r


def depth_synthetic():
    L = lf.synthetic.synthetic_light_field(n_views=7, size=96)
    gt = L.meta["gt_disparity"]
    res = {"defocus": timed("focus", lambda: depth.depth_from_focus(L, SYN_ALPHAS)),
           "correspondence": timed("corr", lambda: depth.depth_from_correspondence(L, SYN_ALPHAS)),
           "EPI tensor": timed("epi", lambda: depth.epi_structure_tensor_disparity(L)),
           "combined": timed("combined", lambda: depth.depth_combined(L, SYN_ALPHAS))}
    fig, ax = plt.subplots(2, 5, figsize=(18, 7.5))
    ax[0, 0].imshow(gt, vmin=-1.5, vmax=2, cmap="viridis"); ax[0, 0].set_title("ground truth")
    ax[1, 0].imshow(np.clip(L.center_view, 0, 1)); ax[1, 0].set_title("centre view")
    print("synthetic metrics (3-px discontinuity band excluded, 8-px border dropped):")
    for j, (k, r) in enumerate(res.items(), 1):
        e = depth.evaluate(r["disparity"], gt, border=8)
        print(f"  {k:15s} MAE {e['mae']:.3f}  RMSE {e['rmse']:.3f}  bad {e['bad_pixel_ratio']:.3f} | all-px MAE {e['mae_all']:.3f}")
        ax[0, j].imshow(r["disparity"], vmin=-1.5, vmax=2, cmap="viridis")
        ax[0, j].set_title(f"{k}\nMAE {e['mae']:.3f} (all {e['mae_all']:.3f})")
        im = ax[1, j].imshow(np.abs(r["disparity"] - gt), vmin=0, vmax=1, cmap="magma")
        ax[1, j].set_title("|error|")
    fig.colorbar(im, ax=ax[1, 4], fraction=0.046)
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); fig.savefig(OUT / "depth_synthetic.png", dpi=110); plt.close(fig)

    # EPI slopes
    row = 50; epi = L.epi_horizontal(row); u0, v0 = L.center
    fig, ax = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [3, 2]})
    ax[0].imshow(np.clip(L.center_view, 0, 1)); ax[0].axhline(row, color="r"); ax[0].axis("off")
    ax[0].set_title(f"centre view, row {row} marked")
    ax[1].imshow(np.clip(epi, 0, 1), aspect=3, interpolation="nearest", extent=[-0.5, L.width - 0.5, L.nv - 0.5, -0.5])
    # pick one pixel on each layer along the row
    cols = {}
    for d in (-1.0, 0.0, 1.5):
        c = np.where(gt[row] == d)[0]
        if len(c): cols[d] = int(c[len(c) // 2])
    vv = np.arange(L.nv)
    for (d, c), col in zip(cols.items(), ["cyan", "yellow", "red"]):
        ax[1].plot(c + d * (vv - v0), vv, color=col, lw=2, label=f"slope dt/dv = {d:+.1f}")
    ax[1].set_xlabel("t (column)"); ax[1].set_ylabel("v"); ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("horizontal EPI: lines through scene points lean by their disparity")
    plt.tight_layout(); fig.savefig(OUT / "epi_slopes.png", dpi=110); plt.close(fig)

    # cost profiles
    stack = refocus.focal_stack(L, SYN_ALPHAS)
    F = depth._focus_volume(depth._gray_stack(stack), "laplacian", 7)
    V = depth._variance_volume(L, SYN_ALPHAS, 5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for (d, c), col in zip(cols.items(), ["tab:green", "tab:orange", "tab:blue"]):
        pass
        ax[0].plot(SYN_ALPHAS, F[:, row, c] / F[:, row, c].max(), color=col, label=f"layer d={d:+.1f}")
        ax[1].plot(SYN_ALPHAS, V[:, row, c] / V[:, row, c].max(), color=col, label=f"layer d={d:+.1f}")
        ax[0].axvline(d, color=col, ls=":"); ax[1].axvline(d, color=col, ls=":")
    ax[0].set_title("focus measure (higher = sharper)"); ax[1].set_title("angular variance (lower = consistent)")
    for a in ax: a.set_xlabel(r"candidate $\alpha$"); a.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(OUT / "cost_volume_profile.png", dpi=110); plt.close(fig)
    return L


def edof_figure(L, alphas, name, focus_alpha, noise=0.02, radius=1):
    full, stopped, aif = edof.compare_with_stop_down(L, alphas, small_radius=radius, noise_sigma=noise,
                                                     focus_alpha=focus_alpha)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, im, t in zip(ax, (full, stopped, aif),
                        (f"full aperture, focus alpha={focus_alpha:g}\n(bright, shallow DoF)",
                         f"stopped down r={radius}\n(deep DoF, noisy)", "light-field all-in-focus\n(bright, deep DoF, clean)")):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=10); a.axis("off")
    plt.tight_layout(); fig.savefig(OUT / f"edof_{name}.png", dpi=110); plt.close(fig)


def real(name):
    L = lf.io.load(name, **REAL_KW[name]); al = REAL_ALPHAS[name]
    print(f"{name}: {L}")
    r = timed("combined", lambda: depth.depth_combined(L, al))
    d = r["disparity"]
    print(f"  disparity percentiles 2/50/98: {np.percentile(d, [2, 50, 98]).round(2)}")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
    ax[0].imshow(np.clip(L.center_view, 0, 1)); ax[0].set_title("centre view")
    im = ax[1].imshow(d, cmap="viridis"); ax[1].set_title("combined disparity"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    im = ax[2].imshow(r["confidence"], vmin=0, vmax=1, cmap="gray"); ax[2].set_title("confidence"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.axis("off")
    plt.tight_layout(); fig.savefig(OUT / f"depth_{name}.png", dpi=110); plt.close(fig)
    lo, hi = np.percentile(d, [10, 90])
    edof_figure(L, al, name, focus_alpha=float(hi), noise=0.02, radius=1)


if __name__ == "__main__":
    L = depth_synthetic()
    edof_figure(L, SYN_ALPHAS, "synthetic", focus_alpha=0.0)
    avail = lf.io.available()
    for n in ("lego", "bracelet", "lytro"):
        if avail.get(n):
            real(n)
        else:
            print(f"{n}: not available, skipped")
    print("figures written to", OUT)
