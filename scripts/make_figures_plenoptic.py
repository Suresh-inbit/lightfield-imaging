#!/usr/bin/env python3
"""Figures for docs/02_plenoptic_camera.md (headless).  Writes to docs/figures/."""
import sys, pathlib, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lf import io, utils, plenoptic as P
from lf.synthetic import synthetic_light_field

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
N = 9
LF = synthetic_light_field(n_views=N, size=128)
RAW = P.to_lenslet(LF)
CLO = (88, 106)      # micro-lens on the edge of the gravel disc (out of focus -> tiles differ)


def save(fig, name):
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


def fig_raw():
    close = P.lenslet_closeup(RAW, (N, N), CLO, 6)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1]})
    ax[0].imshow(RAW); ax[0].set_title(f"simulated raw lenslet image {RAW.shape[:2]}")
    s0, t0 = CLO[0] - 3, CLO[1] - 3
    ax[0].add_patch(plt.Rectangle((t0 * N, s0 * N), 6 * N, 6 * N, fill=False, ec="r", lw=1.5))
    ax[1].imshow(close, interpolation="nearest"); ax[1].set_title("close-up: 6 x 6 micro-lenses (each tile = 9 x 9 views)")
    for k in range(7):
        ax[1].axhline(k * N - 0.5, color="w", lw=0.4); ax[1].axvline(k * N - 0.5, color="w", lw=0.4)
    for a in ax: a.axis("off")
    plt.tight_layout(); save(fig, "lenslet_raw_synthetic.png")


def fig_extract():
    i, j = 2, 6                                  # highlighted pixel under every micro-lens
    close = P.lenslet_closeup(RAW, (N, N), CLO, 6)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 1]})
    ax[0].imshow(close, interpolation="nearest")
    for a in range(6):
        for b in range(6):
            ax[0].add_patch(plt.Rectangle((b * N + j - 0.5, a * N + i - 0.5), 1, 1, fc="none", ec="r", lw=1.5))
            ax[0].add_patch(plt.Rectangle((b * N - 0.5, a * N - 0.5), N, N, fc="none", ec="w", lw=0.5))
    ax[0].set_title(f"same pixel (i,j)=({i},{j}) under each micro-lens")
    ax[1].imshow(LF.views[i, j][CLO[0] - 3:CLO[0] + 3, CLO[1] - 3:CLO[1] + 3], interpolation="nearest")
    ax[1].set_title("= these 6 x 6 pixels of sub-aperture view (u,v)=(2,6)")
    ax[2].imshow(P.from_lenslet(RAW, (N, N)).views[i, j]); ax[2].set_title("full decoded sub-aperture view")
    for a in ax: a.axis("off")
    plt.tight_layout(); save(fig, "subaperture_extraction.png")


def fig_stopdown():
    titles = ["all 81 pixels (conventional photo)", "central disc r=3", "central pixel only (r=0)"]
    radii = [None, 2.5, 0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, r, t in zip(ax, radii, titles):
        a.imshow(np.clip(P.conventional_photo(RAW, (N, N), radius=r), 0, 1)); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); save(fig, "conventional_vs_stopdown.png")


def fig_fnumber():
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for k, n in enumerate([2.8, 4.0, 8.0]):
        raw = P.simulate_fnumber_mismatch(LF, n)
        d = P.design(main_f_number=n)
        ax[0, k].imshow(np.clip(raw, 0, 1), interpolation="nearest")
        ax[0, k].set_title(f"main lens f/{n}: disc = {d['microlens_image_diameter_um']:.0f} um\n"
                           f"overlap {d['overlap_fraction']:.0%}, unlit {d['wasted_fraction']:.0%}", fontsize=10)
        ax[0, k].axis("off")
        ax[1, k].imshow(np.clip(P.lenslet_closeup(raw, (N, N), CLO, 4), 0, 1), interpolation="nearest")
        ax[1, k].axis("off")
    ax[0, 1].set_title(ax[0, 1].get_title() + "  (matched)", fontsize=10)
    plt.tight_layout(); save(fig, "fnumber_matching.png")


def fig_lytro():
    lf = io.load("lytro", crop_angular=None)
    print("lytro", lf)
    m = lf.views.mean(axis=(2, 3, 4))
    fig = utils.view_grid(lf.crop_angular(9), step=1, figsize=(11, 11),
                          title="Lytro Illum flowers: central 9 x 9 sub-aperture views")
    save(fig, "lytro_views_grid.png")
    fig = utils.epi_figure(lf.crop_angular(9), row=lf.height // 2 + 20, figsize=(12, 6))
    save(fig, "lytro_epi.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax[0].imshow(m, cmap="viridis"); plt.colorbar(im, ax=ax[0]); ax[0].set_title("mean intensity per view (14 x 14)")
    ax[0].set_xlabel("v"); ax[0].set_ylabel("u")
    ax[1].plot(m[m.shape[0] // 2], "o-", label="row u=7"); ax[1].plot(m[:, m.shape[1] // 2], "s-", label="column v=7")
    ax[1].set_title("vignetting profile through the centre"); ax[1].set_xlabel("angular index"); ax[1].legend()
    plt.tight_layout(); save(fig, "lytro_vignetting.png")


def fig_lego():
    lf = io.load("lego", downsample=4, angular_crop=9)
    print("lego", lf)
    save(utils.view_grid(lf, step=2, figsize=(9, 9), title="Lego Knights: every 2nd view of the central 9 x 9"),
         "lego_views_grid.png")


if __name__ == "__main__":
    t = time.time()
    fig_raw(); fig_extract(); fig_stopdown(); fig_fnumber()
    av = io.available()
    if av["lytro"]:
        fig_lytro()
    else:
        print("lytro not available, skipped")
    if av["lego"]:
        fig_lego()
    else:
        print("lego not available, skipped")
    print(f"done in {time.time() - t:.1f}s")
