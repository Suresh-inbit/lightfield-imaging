"""Figures for docs 03-05 (refocusing, aperture/DoF, perspective shift). Headless."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import lf
from lf import refocus as R, aperture as A

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)


def show(ax, img, title):
    ax.imshow(np.clip(img, 0, 1).squeeze(), cmap=None if img.shape[-1] == 3 else "gray")
    ax.set_title(title, fontsize=9); ax.axis("off")


def sweep_fig(L, alphas, fname, tag):
    fig, axs = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 3.4))
    for ax, a in zip(axs, alphas):
        show(ax, R.refocus(L, a), f"alpha = {a:+.2f}")
    fig.suptitle(f"Digital refocusing sweep ({tag})"); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=90); plt.close(fig)


def datasets():
    yield "synthetic", lf.synthetic.synthetic_light_field(n_views=9, size=160), {}
    av = lf.io.available()
    if av.get("lego"):
        yield "lego", lf.io.load("lego", downsample=4, angular_crop=9), {}
    if av.get("bracelet"):
        yield "bracelet", lf.io.load("bracelet", downsample=4, angular_crop=9), {}


t0 = time.time()
syn = None
for name, L, _ in datasets():
    if name == "synthetic":
        syn = L
        al = [-1.0, 0.0, 0.75, 1.5, 2.0]
        sweep_fig(L, al, "refocus_sweep_synthetic.png", "synthetic, sharp layers at -1 / 0 / +1.5")
    else:
        lo, hi = R.SUGGESTED_ALPHA_RANGE.get(name, (-1, 1))
        sweep_fig(L, np.linspace(lo, hi, 5), f"refocus_sweep_{name}.png", f"{name}, downsample=4, 9x9 views")
        sw = np.linspace(-2, 2, 41)
        s = R.sweep_sharpness(L, sw)
        print(name, "global sharpness argmax", sw[np.argmax(s)])
    # GIF (only for the first dataset available = synthetic, and real if present)
    al = np.concatenate([np.linspace(-1.5, 2.0, 18), np.linspace(2.0, -1.5, 18)]) if name == "synthetic" \
        else np.concatenate([np.linspace(*R.SUGGESTED_ALPHA_RANGE.get(name, (-1, 1)), 18)] * 2)
    if name == "synthetic":
        imageio.mimsave(OUT / "refocus_sweep.gif", [u8(R.refocus(L, a)) for a in al], duration=0.08, loop=0)
    else:
        imageio.mimsave(OUT / f"refocus_sweep_{name}.gif",
                        [u8(R.refocus(L, a)) for a in list(al[:18]) + list(al[:18][::-1])], duration=0.08, loop=0)

L = syn
# focal stack montage
al = np.linspace(-1.5, 2.0, 8)
fs = R.focal_stack(L, al)
fig, axs = plt.subplots(2, 4, figsize=(14, 8))
for ax, a, im in zip(axs.ravel(), al, fs):
    show(ax, im, f"alpha = {a:+.2f}")
fig.suptitle("Focal stack (synthetic)"); fig.tight_layout()
fig.savefig(OUT / "focal_stack_montage.png", dpi=80); plt.close(fig)

# stop-down comparison (add noise to expose the light-gathering trade-off)
rng = np.random.default_rng(0)
Ln = lf.LightField(np.clip(L.views + rng.normal(0, 0.05, L.views.shape), 0, 1).astype(np.float32), "noisy")
alpha = 0.0
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for j, (r, lab) in enumerate([(1, "radius 1"), (2, "radius 2"), (100, "full aperture")]):
    clean = A.stop_down(L, r, alpha); noisy = A.stop_down(Ln, r, alpha)
    roi = (slice(100, 150), slice(10, 70))   # grass region: out of focus at alpha=0
    ms = np.zeros(L.views.shape[2:4], bool); ms[roi] = True
    show(axs[0, j], clean, f"{lab}, clean: sharpness(grass ROI)={R.sharpness(clean, ms):.2e}")
    show(axs[1, j], noisy, f"{lab}, noise 0.05, averaged over {int(A.aperture_mask(9, 9, 'circle', r).sum())} views")
fig.suptitle("Digital stopping down at alpha=0: small aperture -> grass/gravel sharper (more DoF) but noisier; "
             "full aperture -> shallow DoF, low noise", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "stopdown_comparison.png", dpi=80); plt.close(fig)

# perspective strip: 4 corners + 2 synthesized in-betweens
nu, nv = L.nu, L.nv
pos = [("corner (0,0)", 0, 0), ("corner (0,V-1)", 0, nv - 1), ("synth (2.5, 3.5)", 2.5, 3.5),
       ("synth (5.5, 4.5)", 5.5, 4.5), ("corner (U-1,0)", nu - 1, 0), ("corner (U-1,V-1)", nu - 1, nv - 1)]
fig, axs = plt.subplots(1, 6, figsize=(18, 3.4))
for ax, (t, u, v) in zip(axs, pos):
    show(ax, A.synthesize_view(L, u, v), t)
fig.suptitle("Viewpoint change: gravel (near) moves most, grass (far) moves opposite; synthesized views show mild ghosting",
             fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "perspective_shift_strip.png", dpi=80); plt.close(fig)
imageio.mimsave(OUT / "perspective_shift.gif", [u8(x) for x in A.perspective_sequence(L, "circle", 24)],
                duration=0.08, loop=0)
print("done in %.1fs" % (time.time() - t0))
