"""Generate the five guided light-field notebooks in this folder.

Run from the project root:  python notebooks/_build_notebooks.py
Each notebook is a list of (kind, source) cells; kind is "md" or "code".
The notebooks are then executed with nbconvert (see README for the command).
"""
from __future__ import annotations
import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = pathlib.Path(__file__).resolve().parent

HEADER = r"""%matplotlib inline
# JPEG keeps the executed notebooks small (plots are photographs of textures).
%config InlineBackend.figure_format = 'jpeg'
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 80
# Make `import lf` work whether Jupyter was started in the project root or in notebooks/.
_cwd = pathlib.Path.cwd().resolve()
ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


def build(filename: str, cells):
    nb = new_notebook()
    nb.cells = []
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, str(HERE / filename))
    print("wrote", HERE / filename, len(nb.cells), "cells")


# =============================================================================
# 01  Light field basics
# =============================================================================
NB01 = [
    md(r"""
# 01. Light field basics
"""),
    md(r"""
A normal photo adds up all the rays that reach each pixel, so the direction the light came from is lost at the moment of capture. A light field keeps that direction: $L(u, v, s, t)$ stores one small image for every position $(u, v)$ on the lens. With that 4-D array, changing the viewpoint, the focus and the depth of field become computations you can do after the shutter. This notebook looks at a synthetic scene and the Stanford Lego Knights gantry capture, builds epipolar images, and shows how a plenoptic camera packs the same 4-D data into one sensor image.
"""),
    code(HEADER + r"""
import lf
from lf import io as lfio, plenoptic, utils, synthetic

print(lfio.available())
"""),
    md(r"""
## A synthetic light field

The synthetic scene has three textured layers at known disparities (grass at $-1$, text at $0$, gravel at $+1.5$ px per view step), so every later result can be checked against `meta['gt_disparity']`. The array is `(U, V, H, W, C)`: angular rows, angular columns, image rows, image columns, colour.
"""),
    code(r"""
L_syn = synthetic.synthetic_light_field(n_views=7, size=128)
print(L_syn)
print('views shape (U, V, H, W, C):', L_syn.views.shape, '| dtype:', L_syn.views.dtype)
print('layer disparities:', [l['disparity'] for l in L_syn.meta['layers']])
utils.view_grid(L_syn, step=1, figsize=(9, 9), title='Synthetic scene: all 7 x 7 sub-aperture views');
"""),
    md(r"""
Neighbouring views look almost identical, but the near gravel disc moves much more between views than the far grass does. That relative motion is the parallax that refocusing and depth estimation exploit in the next notebooks.
"""),
    md(r"""
## Lego Knights, a real gantry capture

The Stanford gantry moved a camera over a plane. We keep the central $9 \times 9$ views and downsample by 4, which keeps the data at about 60 MB. Never lower `downsample` for this dataset: the full-resolution set is 3.6 GB as float32.
"""),
    code(r"""
L_lego = lfio.load('lego', downsample=4, angular_crop=9)
print(L_lego, '| memory:', round(L_lego.views.nbytes / 1e6), 'MB')
utils.view_grid(L_lego, step=2, figsize=(9, 9), title='Lego Knights: every second view of the 9 x 9 grid');
"""),
    md(r"""
## Epipolar-plane images (EPIs)

Fix one image row and one angular row, then stack that image row across all angular columns. The result is an EPI. A scene point with disparity $d$ shifts by $d$ pixels for each step of the viewpoint, so it traces a straight line whose slope is $d$. The right panel is the EPI of row 70, which crosses all three layers of the synthetic scene.
"""),
    code(r"""
utils.epi_figure(L_syn, row=70, figsize=(10, 6));
"""),
    md(r"""
Each textured stripe in the EPI is a straight line, and its tilt is the disparity of that layer. The text sits on the focal plane, so its line is vertical; the grass and the gravel tilt in opposite directions, with the nearer gravel tilting more.
"""),
    md(r"""
## A real micro-lens capture (optional)

If the Lytro Illum ESLF is on disk, decode it into the same `(U, V, H, W, C)` layout. Its 14 x 14 micro-lens tiles were already split into sub-aperture images by `lf.io`. Otherwise this cell only prints a message.
"""),
    code(r"""
if lfio.available()['lytro']:
    L_real = lfio.load('lytro', downsample=2)
    print(L_real)
    utils.view_grid(L_real, step=4, figsize=(7, 7), title='Real Lytro Illum capture (decoded ESLF, 3 x 3 of 9 x 9 views)')
else:
    print('Lytro dataset not on disk; run scripts/download_data.py lytro to enable this cell.')
"""),
    md(r"""
## The plenoptic sensor image

A plenoptic camera puts a micro-lens array in front of the sensor, so each micro-lens covers a block of pixels that records the rays arriving from different parts of the lens. `to_lenslet` simulates that raw mosaic and `from_lenslet` inverts it exactly. Here we use the synthetic scene because its raw image is small enough to display.
"""),
    code(r"""
raw = plenoptic.to_lenslet(L_syn)
print('raw sensor image:', raw.shape, '| light field:', L_syn.views.shape)
print('round trip is exact:', np.allclose(plenoptic.from_lenslet(raw, (7, 7)).views, L_syn.views))

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6))
a.imshow(np.clip(raw, 0, 1))
a.set_title('simulated raw sensor (7 x 7 pixels per micro-lens)'); a.axis('off')
close = plenoptic.lenslet_closeup(raw, (7, 7), center_st=(64, 64), n_lenses=6)
b.imshow(np.clip(close, 0, 1))
b.set_title('6 x 6 micro-lens close-up'); b.axis('off')
plt.tight_layout()
"""),
    md(r"""
Each tile is one micro-lens, and the pixels inside a tile are the rays that micro-lens received, one pixel per angular position. Resolution is traded for angle: here 7 x 7 angular samples cost a factor 49 in spatial pixels.
"""),
    md(r"""
## Sub-aperture extraction

Taking the same pixel under every micro-lens gives one sub-aperture image, which is the view through one part of the lens. `from_lenslet` does this for all $(u, v)$ at once.
"""),
    code(r"""
L_dec = plenoptic.from_lenslet(raw, (7, 7))
print('raw[1::7, 2::7] equals view (1, 2):', np.allclose(raw[1::7, 2::7], L_dec.view(1, 2)))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (u, v) in zip(axes, [(3, 3), (3, 0), (0, 3)]):
    ax.imshow(np.clip(L_dec.view(u, v), 0, 1))
    ax.set_title(f'sub-aperture view ({u}, {v})'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The centre view is the pinhole picture through the middle of the lens, and the edge views see the scene from the side. The shift between them is exactly the parallax from the EPI section.
"""),
    md(r"""
## One exposure, many photographs

A conventional photo is the average of all pixels under each micro-lens. Averaging only a central disc of the block is the same as a smaller aperture, so one plenoptic capture gives a whole family of stopped-down photos after the fact. The first panel is the full aperture; the others use discs of radius 4, 2, 1 and 0 views.
"""),
    code(r"""
panels = [('full aperture', None), ('radius 4', 4), ('radius 2', 2), ('radius 1', 1), ('radius 0 (one view)', 0)]
fig, axes = plt.subplots(1, 5, figsize=(17, 4))
for ax, (title, r) in zip(axes, panels):
    img = plenoptic.conventional_photo(L_lego, radius=r)
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
With the full aperture only the rectification plane is sharp, and the blur grows away from it. Shrinking the disc buys depth of field, until a single view is sharp everywhere. This data is noise-free, so the cost of stopping down (less light, more noise) does not show here.
"""),
    md(r"""
## Designing a plenoptic camera

`plenoptic.design` turns sensor size, pixel pitch, micro-lens pitch and main-lens f-number into the numbers of Ng et al. (2005). The micro-lens f-number is $f_m / p$, and it should equal the main-lens f-number, otherwise the micro-lens images are the wrong size for their cells.
"""),
    code(r"""
designs = {f'f/{N}': plenoptic.design(main_f_number=N) for N in (4.0, 2.8)}
keys = ['angular_resolution', 'n_microlenses', 'microlens_f_number', 'f_numbers_match',
        'microlens_image_diameter_px', 'overlap_fraction', 'wasted_fraction', 'total_rays']
print(f"{'quantity':30s}" + ''.join(f'{name:>16s}' for name in designs))
for k in keys:
    cells = []
    for d in designs.values():
        val = d[k]
        cells.append(f'{val:16.3f}' if isinstance(val, float) else f'{str(val):>16s}')
    print(f'{k:30s}' + ''.join(cells))

print()
print(f"{'camera':60s} {'sensor':>12s} {'angular':>9s} {'spatial (slide)':>16s}")
for row in plenoptic.camera_table():
    print(f"{row['camera']:60s} {str(row['sensor_px']):>12s} {str(row['angular']):>9s} {str(row['spatial_slide']):>16s}")
"""),
    md(r"""
The f/4 column matches: the micro-lens images just fill their cells, with little overlap. The f/2.8 main lens overfills the micro-lenses, so neighbouring cells overlap and cross-talk rises. The last table is the resolution trade-off from lecture 16.
"""),
    md(r"""
## Where to go next

Continue with [docs/02_plenoptic_camera.md](../docs/02_plenoptic_camera.md) for the derivations behind this notebook, and read [docs/00_background.md](../docs/00_background.md) and [docs/01_datasets.md](../docs/01_datasets.md) if you want the notation and the data formats. Notebook 02 uses the same light fields to refocus and change the aperture. The matching tab in the Streamlit explorer (`streamlit run app/streamlit_app.py`) lets you play with the same functions.
"""),
]

# =============================================================================
# 02  Refocus, aperture, perspective
# =============================================================================
NB02 = [
    md(r"""
# 02. Refocusing, aperture and perspective
"""),
    md(r"""
Once a light field is captured, focus is a computation: shift each view in proportion to its position on the lens and average them. Objects whose disparity matches the shift line up and stay sharp, and everything else blurs in the same way a real defocused lens would blur it. The same sum over a subset of views is a digital aperture, and interpolating between views gives a new viewpoint. This notebook refocuses the Lego Knights and the synthetic scene, measures where each plane is sharpest, and then stops down, moves the viewpoint and renders a perspective sequence.
"""),
    code(HEADER + r"""
import time
from ipywidgets import interact, FloatSlider
import lf
from lf import io as lfio, refocus, aperture, synthetic, utils

L_lego = lfio.load('lego', downsample=4, angular_crop=9)
L_syn = synthetic.synthetic_light_field(n_views=7, size=128)
print(L_lego)
print(L_syn)
"""),
    md(r"""
## Refocusing with alpha

The focus parameter $\alpha$ is the disparity of the plane you want sharp, in pixels per view step. On the synthetic scene the grass is sharp at $\alpha=-1$, the text at $\alpha=0$ and the gravel at $\alpha=1.5$; in between, nothing is sharp.
"""),
    code(r"""
alphas_syn = [-1.0, 0.0, 0.75, 1.5]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, a in zip(axes, alphas_syn):
    ax.imshow(np.clip(refocus.refocus(L_syn, a), 0, 1))
    ax.set_title(f'alpha = {a:+.2f}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
Each layer comes into focus only at its own disparity, which is why the refocusing parameter doubles as a depth estimate. The image at $\alpha = 0.75$ is blurred everywhere because no layer has that disparity.
"""),
    code(r"""
alphas_lego = [-1.0, 0.0, 0.5]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, a in zip(axes, alphas_lego):
    ax.imshow(np.clip(refocus.refocus(L_lego, a), 0, 1))
    ax.set_title(f'Lego, alpha = {a:+.2f}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
At $\alpha = -1$ the castle wall is sharp and the knights are blurred; at $\alpha$ between 0 and 0.5 the knights come into focus instead. These are the ranges that `refocus.SUGGESTED_ALPHA_RANGE` suggests for this crop.
"""),
    md(r"""
The cell shows a static preview at the default value first, which is also what a headless run records. In JupyterLab the slider then takes over and refocuses interactively.
"""),
    code(r"""
def show_refocus(alpha=0.0):
    img = refocus.refocus(L_lego, alpha)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'Lego Knights, alpha = {alpha:+.2f}'); ax.axis('off')
    plt.tight_layout()
    plt.show()

show_refocus(0.0)   # static preview (kept in the saved notebook)
interact(show_refocus, alpha=FloatSlider(min=-2.0, max=1.5, step=0.05, value=0.0,
                                         continuous_update=False, description='alpha'));
"""),
    md(r"""
## How sharp is each plane?

`sweep_sharpness` refocuses over a range of $\alpha$ and scores a region of interest with the variance of the Laplacian. Plotting the score against $\alpha$ shows the focus peak for each region. We use one region on the text plane and one on the gravel disc of the synthetic scene.
"""),
    code(r"""
alphas = np.linspace(-1.5, 2.0, 36)
text_roi = (slice(30, 60), slice(30, 60))
gravel_roi = (slice(75, 95), slice(75, 95))
s_text = refocus.sweep_sharpness(L_syn, alphas, roi=text_roi)
s_gravel = refocus.sweep_sharpness(L_syn, alphas, roi=gravel_roi)
print('text ROI peaks at alpha   =', round(float(alphas[np.argmax(s_text)]), 2))
print('gravel ROI peaks at alpha =', round(float(alphas[np.argmax(s_gravel)]), 2))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(alphas, s_text / s_text.max(), label='text ROI (true d = 0)')
ax.plot(alphas, s_gravel / s_gravel.max(), label='gravel ROI (true d = 1.5)')
ax.set_xlabel('alpha'); ax.set_ylabel('sharpness (normalised)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
"""),
    md(r"""
Each region peaks at its own disparity, so the sweep recovers depth without any calibration. The peaks are slightly broad because a bilinear shift is a weak low-pass filter at fractional $\alpha$.
"""),
    md(r"""
## Three ways to shift a view

`refocus` supports three shift methods: `subpixel` (bilinear, replicated borders), `integer` (each view rounded to whole pixels) and `fourier` (a phase ramp, exact for band-limited images but periodic at the borders). We compare them at a fractional $\alpha = 0.5$, using `fourier` as the reference and ignoring a 8-pixel border where the boundary models differ.
"""),
    code(r"""
a = 0.5
out, times = {}, {}
for m in ['subpixel', 'integer', 'fourier']:
    t0 = time.perf_counter()
    out[m] = refocus.refocus(L_syn, a, method=m)
    times[m] = 1e3 * (time.perf_counter() - t0)

ref = out['fourier']
print(f"{'method':10s} {'time [ms]':>10s} {'PSNR vs fourier (interior)':>28s}")
for m in out:
    p = utils.psnr(out[m][8:-8, 8:-8], ref[8:-8, 8:-8])
    print(f"{m:10s} {times[m]:10.1f} {p:28.1f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, m in zip(axes, out):
    ax.imshow(np.clip(out[m], 0, 1)); ax.set_title(f'{m}, alpha = {a}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The subpixel result is close to the Fourier reference (about 52 dB in the interior), and integer rounding is clearly further off (about 42 dB). Fourier is the most accurate and the slowest of the three, and the borders differ most, where replicated and periodic boundaries disagree.
"""),
    md(r"""
## A focal stack

Refocusing at a set of $\alpha$ values gives a focal stack. Each frame of the Lego montage is focused on a different depth, so the wall and the knights take turns being sharp.
"""),
    code(r"""
stack_alphas = np.linspace(-2.0, 1.5, 8)
stack = refocus.focal_stack(L_lego, stack_alphas)
print('focal stack shape:', stack.shape)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, a, im in zip(axes.ravel(), stack_alphas, stack):
    ax.imshow(np.clip(im, 0, 1)); ax.set_title(f'alpha = {a:+.2f}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The focal stack is the input to the all-in-focus and depth-from-focus methods of notebook 03. Every frame throws away some of the light that the others keep.
"""),
    md(r"""
## Digital stopping down

`stop_down` refocuses using only the views inside a disc of a given radius, which is an aperture you choose after the exposure. `effective_f_number` converts that radius to the f-number of an equivalent physical camera, using the digital refocusing theorem with a main lens of f/4 and 9 pixels per micro-lens. The panels below are all focused on the castle wall at $\alpha = -1$.
"""),
    code(r"""
radii = [4, 2, 1, 0]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, r in zip(axes, radii):
    img = aperture.stop_down(L_lego, radius=r, alpha=-1.0)
    n_views = int(aperture.aperture_mask(9, 9, 'circle', r).sum())
    fnum = aperture.effective_f_number(4.0, 9, r)
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(f'radius {r}: {n_views} views, f/{fnum:.1f}')
    ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
A smaller radius gives a larger depth of field: at radius 0 one view is a pinhole image and nothing is blurred by averaging. The price is light, which is free in this noise-free data but costs SNR on a real sensor.
"""),
    md(r"""
## Fractional viewpoints

`synthesize_view` interpolates between the four nearest views at a fractional position $(u, v)$. The midpoint between four views is a plausible new viewpoint for distant things, but nearby things with larger disparity appear twice.
"""),
    code(r"""
u0, v0 = L_lego.center
panels = [(f'view ({u0}, {v0})', L_lego.view(u0, v0)),
          (f'view ({u0}, {v0 + 1})', L_lego.view(u0, v0 + 1)),
          (f'synthesized ({u0 + 0.5}, {v0 + 0.5})', aperture.synthesize_view(L_lego, u0 + 0.5, v0 + 0.5))]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The synthesized midpoint is a plausible blend of its neighbours at this small step. Doubled edges appear when disparities are larger than about a pixel, and avoiding them needs a depth map to warp each neighbour before blending.
"""),
    md(r"""
## Moving the observer

`perspective_sequence` walks the viewpoint around a circle inside the aperture. The strip below shows eight frames of that path; playing the full 24-frame sequence gives a parallax animation.
"""),
    code(r"""
frames = aperture.perspective_sequence(L_lego, path='circle', n=8)
fig, axes = plt.subplots(1, 8, figsize=(16, 2.8))
for k, (ax, f) in enumerate(zip(axes, frames)):
    ax.imshow(np.clip(f, 0, 1)); ax.set_title(f'{k}', fontsize=9); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The background shifts relative to the knights from frame to frame, and that relative motion is the parallax. The sequence is closed, so it loops cleanly.
"""),
    md(r"""
As before, a static preview at the default viewpoint is shown first, and the sliders then move the viewpoint in JupyterLab.
"""),
    code(r"""
def show_viewpoint(u=4.0, v=4.0):
    img = aperture.synthesize_view(L_lego, u, v)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f'viewpoint (u, v) = ({u:.2f}, {v:.2f})'); ax.axis('off')
    plt.tight_layout()
    plt.show()

show_viewpoint(4.0, 4.0)   # static preview (kept in the saved notebook)
interact(show_viewpoint,
         u=FloatSlider(min=0.0, max=8.0, step=0.1, value=4.0, continuous_update=False),
         v=FloatSlider(min=0.0, max=8.0, step=0.1, value=4.0, continuous_update=False));
"""),
    md(r"""
## Where to go next

Read [docs/03_refocusing.md](../docs/03_refocusing.md) for the shift-and-add derivation and the Fourier and integer variants, [docs/04_aperture_dof.md](../docs/04_aperture_dof.md) for the digital refocusing theorem, and [docs/05_perspective_shift.md](../docs/05_perspective_shift.md) for view synthesis. Notebook 03 uses the focal stack to estimate depth and to build an all-in-focus image. The matching tabs in the Streamlit explorer (`streamlit run app/streamlit_app.py`) expose the same sliders.
"""),
]

# =============================================================================
# 03  Depth and extended DoF
# =============================================================================
NB03 = [
    md(r"""
# 03. Depth from a light field and extended depth of field
"""),
    md(r"""
A point's depth is encoded in how it moves between views, and that movement shows up in two places: the $\alpha$ at which it is sharpest after refocusing, and the slope of its line in an epipolar image. This notebook runs four estimators: defocus (sharpest plane), correspondence (views that agree), the EPI slope from a structure tensor, and a fused version of the first two. The synthetic scene comes with true disparity, so we can score each method with MAE, RMSE and bad-pixel ratio. We then run the same estimators on Lego Knights, which has no ground truth, and finish with an all-in-focus image from a focal stack.
"""),
    code(HEADER + r"""
import lf
from lf import io as lfio, synthetic, depth, edof, utils

L_syn = synthetic.synthetic_light_field(n_views=7, size=128)
gt = L_syn.meta['gt_disparity']
alphas_syn = np.linspace(-1.5, 2.0, 15)
print(L_syn, '| ground-truth disparity range:', L_syn.meta['disparity_range'])
"""),
    md(r"""
## Scoring the estimators on the synthetic scene

Each estimator returns a disparity map. `depth.evaluate` compares it with the ground truth after ignoring an 8-pixel border, and by default also ignores a 3-pixel band around depth edges, where the answer is ambiguous. The `_all` numbers include the edges.
"""),
    code(r"""
estimates = {
    'defocus': depth.depth_from_focus(L_syn, alphas_syn),
    'correspondence': depth.depth_from_correspondence(L_syn, alphas_syn),
    'EPI slope': depth.epi_structure_tensor_disparity(L_syn),
    'combined': depth.depth_combined(L_syn, alphas_syn),
}
print(f"{'method':16s} {'MAE':>7s} {'RMSE':>7s} {'bad (edges excl.)':>18s} {'bad (all px)':>13s}")
for name, r in estimates.items():
    m = depth.evaluate(r['disparity'], gt, border=8)
    print(f"{name:16s} {m['mae']:7.3f} {m['rmse']:7.3f} {m['bad_pixel_ratio']:18.3f} {m['bad_pixel_ratio_all']:13.3f}")
"""),
    code(r"""
items = [('ground truth', gt)] + [(k, r['disparity']) for k, r in estimates.items()]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
for ax, (title, d) in zip(axes, items):
    im = ax.imshow(d, cmap='viridis', vmin=-1.0, vmax=1.5)
    ax.set_title(title); ax.axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label='disparity (px per view step)')
"""),
    md(r"""
Defocus is the most accurate estimator on this scene (MAE 0.009), with EPI slope second. The combined estimator is not better than defocus alone here: the correspondence cue pulls it down on the grass (see docs/07), and its all-pixel bad-pixel ratio is the highest of the four (0.22). Errors cluster at the layer boundaries, which the evaluation excludes from the bad-pixel band but the all-pixel column counts.
"""),
    md(r"""
## Lego Knights, no ground truth

On real data we can only check the maps by eye. The wall should sit near disparity $-1$ and the knights between $0$ and $0.5$, with the same range that `refocus.SUGGESTED_ALPHA_RANGE` gives for this crop.
"""),
    code(r"""
L_lego = lfio.load('lego', downsample=4, angular_crop=9)
alphas_lego = np.linspace(-2.0, 1.5, 15)
lego_est = {
    'defocus': depth.depth_from_focus(L_lego, alphas_lego),
    'correspondence': depth.depth_from_correspondence(L_lego, alphas_lego),
    'EPI slope': depth.epi_structure_tensor_disparity(L_lego),
    'combined': depth.depth_combined(L_lego, alphas_lego),
}
items = [('centre view', L_lego.center_view)] + [(k, r['disparity']) for k, r in lego_est.items()]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
for ax, (title, d) in zip(axes, items):
    if title == 'centre view':
        ax.imshow(np.clip(d, 0, 1))
    else:
        im = ax.imshow(d, cmap='viridis', vmin=-1.2, vmax=0.8)
    ax.set_title(title); ax.axis('off')
fig.colorbar(im, ax=axes[1:].tolist(), shrink=0.85, label='disparity (px per view step)')
"""),
    md(r"""
The correspondence and EPI maps keep the wall dark (near $-1$) and the knights in the middle range. Defocus is the most erratic: the brick texture creates false sharpness peaks on the wall, and the combined map is the smoothest but softens the knights' outlines.
"""),
    md(r"""
The estimator confidence tells you where to trust a map. Here is the combined confidence for the same scene.
"""),
    code(r"""
conf = depth.depth_combined(L_lego, alphas_lego)['confidence']
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(conf, cmap='magma', vmin=0, vmax=1)
ax.set_title('combined confidence'); ax.axis('off')
fig.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout()
"""),
    md(r"""
Confidence is lowest along the knights' edges and at some brick joints, where the two cues disagree, and it is high on the flat baseplate and most of the wall. The fusion step weights each cue by this map.
"""),
    md(r"""
## Bracelet (optional)

The Stanford bracelet is another gantry capture with a different depth range. If it is on disk, the cell below runs the combined estimator on it. Otherwise it prints a message.
"""),
    code(r"""
if lfio.available()['bracelet']:
    L_br = lfio.load('bracelet', downsample=4, angular_crop=9)
    d_br = depth.depth_combined(L_br, np.linspace(-2.0, 1.5, 15))['disparity']
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(d_br, cmap='viridis'); ax.set_title('bracelet, combined disparity'); ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
else:
    print('Bracelet is not on disk; run scripts/download_data.py bracelet to enable this cell.')
"""),
    md(r"""
## Extended depth of field

A focal stack gives an all-in-focus image: for each pixel, take the frame where it is sharpest, with soft weights so the transitions are smooth. `compare_with_stop_down` puts three images side by side: the full aperture focused on the wall, the same lens stopped down to a small disc, and the all-in-focus merge of the full light field. The stopped-down image gets synthetic noise, scaled by the lost light.
"""),
    code(r"""
alphas_edof = np.linspace(-2.0, 1.5, 15)
full, stopped, aif = edof.compare_with_stop_down(L_lego, alphas_edof, small_radius=1,
                                                 noise_sigma=0.02, focus_alpha=-1.0)
panels = [('full aperture, focused at -1', full),
          ('stopped down (radius 1)', stopped),
          ('all-in-focus from focal stack', aif)]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The full aperture is smooth, and only the wall is sharp. Stopping down brings the knights into focus, but the image is grainy because the simulated noise is scaled by the light that is lost. The all-in-focus result keeps the full light and is sharp at both depths, though it may show halos where depth changes abruptly.
"""),
    md(r"""
## Where to go next

Read [docs/06_extended_dof.md](../docs/06_extended_dof.md) for the weighting scheme and its limits, and [docs/07_depth_estimation.md](../docs/07_depth_estimation.md) for the four estimators, the confidence measures and the evaluation protocol. Notebook 04 moves from the software aperture to a programmable aperture in the camera. The matching tabs in the Streamlit explorer (`streamlit run app/streamlit_app.py`) show the depth maps with sliders.
"""),
]

# =============================================================================
# 04  Programmable aperture
# =============================================================================
NB04 = [
    md(r"""
# 04. Programmable aperture and multiplexed capture
"""),
    md(r"""
A programmable aperture opens a different set of cells in each exposure, so one camera can record every view of the light field over time. If each shot opens about half the cells, it collects far more light than a shot through one cell, and the views can be recovered with a known matrix inverse. Whether this helps depends on the noise, and the slide-29 comparison with the plenoptic camera is where that question comes up. This notebook builds the S-matrix patterns, optimises patterns for nine cells, simulates noisy captures, demultiplexes them, compares the noise with the closed forms, and refocuses the recovered light field.
"""),
    code(HEADER + r"""
import lf
from lf import synthetic, multiplex as mx, refocus, utils

np.set_printoptions(precision=3, suppress=True)
"""),
    md(r"""
## S-matrix algebra

For $n = 7$, the S-matrix $S$ is a 0/1 matrix with $(n+1)/2$ open cells per row, and it satisfies $S^\top S = \frac{n+1}{4}(I + J)$ and $S^{-1} = \frac{2}{n+1}(2S^\top - J)$, where $J$ is the all-ones matrix. We check both identities numerically, and the trace of $(S^\top S)^{-1}$, which sets the read-noise penalty.
"""),
    code(r"""
n = 7
S = mx.s_matrix(n)
print(S.astype(int))
print('open cells per row :', S.sum(1).astype(int))
J = np.ones((n, n)); I = np.eye(n)
print('S^T S == (n+1)/4 (I + J):', np.allclose(S.T @ S, (n + 1) / 4 * (I + J)))
print('S^-1 == 2/(n+1) (2 S^T - J):', np.allclose(np.linalg.inv(S), 2 / (n + 1) * (2 * S.T - J)))
print('tr((S^T S)^-1) =', round(float(np.trace(np.linalg.inv(S.T @ S))), 4),
      '| 4 n^2 / (n+1)^2 =', round(4 * n**2 / (n + 1) ** 2, 4))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(S, cmap='gray'); axes[0].set_title('S-matrix, n = 7'); axes[0].axis('off')
axes[1].imshow(mx.s_matrix(15), cmap='gray'); axes[1].set_title('S-matrix, n = 15'); axes[1].axis('off')
plt.tight_layout()
"""),
    md(r"""
Both identities hold exactly, so the S-matrix inverse is cheap and the 7-cell design has the read-noise factor $4n^2/(n+1)^2 \approx 3.06$ instead of $n = 7$ for sequential capture.
"""),
    md(r"""
## Patterns for nine cells

No S-matrix exists for $n = 9$, so `optimize_patterns` runs a simulated-annealing search on the read-noise objective $\mathrm{tr}((W^\top W)^{-1})$. We use a short run here; the result is a 9 x 9 pattern matrix with one row per shot.
"""),
    code(r"""
W_opt, obj = mx.optimize_patterns(9, iters=5000, noise='read', rng=0)
print('open fraction:', round(float(W_opt.mean()), 2))
print('objective tr((W^T W)^-1): optimised =', round(obj, 3),
      '| identity =', round(float(np.trace(np.linalg.inv(np.eye(9)))), 3))
fig, ax = plt.subplots(figsize=(4.5, 4.2))
ax.imshow(W_opt, cmap='gray', interpolation='nearest')
ax.set_title('W: shots (rows) x cells (columns)'); ax.set_xlabel('cell j'); ax.set_ylabel('shot k')
plt.tight_layout()
"""),
    code(r"""
L = synthetic.synthetic_light_field(n_views=3, size=96)
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for k, ax in enumerate(axes.ravel()):
    ap = mx.pattern_to_aperture(W_opt[k], (3, 3))
    ax.imshow(ap, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'shot {k}', fontsize=9); ax.axis('off')
fig.suptitle('Aperture opened in each of the 9 shots (3 x 3 cells, white = open)')
plt.tight_layout()
"""),
    md(r"""
Each shot opens about half the cells, in a pattern that looks random but is chosen to minimise the read-noise objective. Sequential capture would instead open one cell per shot.
"""),
    md(r"""
## Capture with noise, then demultiplex

We simulate both designs with the same total exposure $T = 1$. The sequential camera opens one cell per shot, so each shot has $\tau = T/9$; the multiplexed camera opens about half the cells, but each shot still has $\tau = T/9$. `demultiplex` applies $W^+$ to the shots and rescales by $\tau$.
"""),
    code(r"""
Y_mux = mx.capture(L, W_opt, total_exposure=1.0, photon_scale=1000, read_noise=0.01, rng=0)
Y_seq = mx.capture(L, np.eye(9), total_exposure=1.0, photon_scale=1000, read_noise=0.01, rng=0)
est_mux = mx.demultiplex(Y_mux, W_opt, exposure_per_shot=1.0 / 9)
est_seq = mx.demultiplex(Y_seq, np.eye(9), exposure_per_shot=1.0 / 9)

print(f"{'scheme':26s} {'PSNR vs clean [dB]':>20s}")
for name, est in [('sequential (identity)', est_seq), ('multiplexed (optimised)', est_mux)]:
    print(f"{name:26s} {utils.psnr(est.views, L.views):20.2f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
panels = [('one captured shot (multiplexed)', Y_mux[0]), ('clean centre view', L.center_view),
          ('sequential, centre view', est_seq.center_view), ('multiplexed, centre view', est_mux.center_view)]
for ax, (title, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title, fontsize=10); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
At the same total exposure, the multiplexed reconstruction is only slightly better in PSNR here (19.7 against 19.2 dB), in this small 3 x 3 example with photon noise. Each multiplexed shot is much brighter than a sequential shot, but the inversion mixes the noise back in, so the net gain under photon noise is small.
"""),
    md(r"""
## Expected noise versus simulation

`expected_mse` gives the closed-form per-sample MSE for read noise and for photon noise. `simulate_snr` runs the full capture and demultiplex to measure the same quantity. We check the two regimes separately and include the plenoptic baseline, which records every sample for the full $T$.
"""),
    code(r"""
x_bar = L.views.reshape(9, -1).mean(1)
cases = [('sequential (identity)', np.eye(9), 1 / 9, False),
         ('multiplexed (optimised)', W_opt, 1 / 9, False),
         ('plenoptic baseline', np.eye(9), 1.0, True)]
print(f"{'scheme':24s} {'read: expected':>15s} {'read: simulated':>16s} {'photon: expected':>17s} {'photon: simulated':>18s}")
for name, W, tau, pl in cases:
    e_r = mx.expected_mse(W, noise='read', sigma_r=0.01, exposure=tau)
    s_r = mx.simulate_snr(L, W, total_exposure=1.0, photon_scale=None, read_noise=0.01,
                          trials=20, rng=1, plenoptic=pl)['mse']
    e_p = mx.expected_mse(W, noise='photon', x_mean=x_bar, exposure=tau, photon_scale=1000.0)
    s_p = mx.simulate_snr(L, W, total_exposure=1.0, photon_scale=1000.0, read_noise=0.0,
                          trials=20, rng=1, plenoptic=pl)['mse']
    print(f"{name:24s} {e_r:15.2e} {s_r:16.2e} {e_p:17.2e} {s_p:18.2e}")
print('S-matrix SNR gain over sequential (read noise, n = 9 optimised W):',
      round(mx.snr_gain(W_opt, noise='read'), 2))
"""),
    md(r"""
The expected and simulated MSEs agree to within about 1 % in every row, which validates the closed forms. Under read noise the multiplexed scheme has about 2.4 times lower MSE than sequential capture, while under photon noise it has about 1.8 times higher MSE, which is the trade-off the next section explains. The plenoptic baseline has the lowest MSE in both regimes, but it ignores its loss of spatial resolution.
"""),
    md(r"""
## Slide 29, revisited

Slide 29 quotes multiplexed SNR as $\mathrm{SNR}_p/\sqrt{2}$ under photon noise and $\mathrm{SNR}_p/2$ under read noise, but the sequential column uses a per-pixel convention, so the two columns are not directly comparable. Under one consistent convention, multiplexing helps mainly with read noise, by about $\sqrt{N}/2$ over sequential capture, and it is about $\sqrt{2}$ worse than sequential for photon noise on a uniform scene (1.35 times for the optimised 9-cell pattern here; see [docs/08](../docs/08_programmable_aperture.md), section 2.4).
"""),
    code(r"""
tt = mx.theory_table(9, W=W_opt)
print(f"{'regime':8s} {'scheme':12s} {'slide claim':>12s} {'per pixel':>11s} {'binned to M/N':>14s}")
for regime in ['photon', 'read']:
    slide = tt['slide'][regime]
    for key, label, s_val in [('pa_nomux', 'no mux', slide[1]), ('pa_mux', 'mux', slide[2])]:
        row = tt[regime][key]
        print(f"{regime:8s} {label:12s} {s_val:12.3f} {row['snr_rel']:11.3f} {row['snr_rel_binned']:14.3f}")
print('plenoptic reference: SNR relative to itself = 1 in every row')
"""),
    md(r"""
The table prints the closed forms in the three columns that matter: the slide's own claim, the per-pixel SNR relative to the plenoptic sample, and the same after binning $N$ pixels to get $M/N$ outputs. Only the binned column is directly comparable with the slide's multiplexed row.
"""),
    md(r"""
## Refocusing the recovered light field

The demultiplexed light fields are ordinary `LightField` objects, so refocusing works on them unchanged. We refocus the clean light field, the sequential reconstruction and the multiplexed reconstruction on the gravel plane at $\alpha = 1.5$, and score each against the clean image.
"""),
    code(r"""
alpha = 1.5
ref_clean = refocus.refocus(L, alpha)
ref_seq = refocus.refocus(est_seq, alpha)
ref_mux = refocus.refocus(est_mux, alpha)
print(f"{'source':26s} {'PSNR vs clean refocus [dB]':>27s}")
print(f"{'sequential':26s} {utils.psnr(ref_seq, ref_clean):27.2f}")
print(f"{'multiplexed':26s} {utils.psnr(ref_mux, ref_clean):27.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, img) in zip(axes, [('clean', ref_clean), ('sequential', ref_seq), ('multiplexed', ref_mux)]):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(f'{title}, alpha = {alpha}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
Both refocus well, and the multiplexed one is slightly closer to the clean refocus (32.5 against 32.1 dB), the same ranking as before.
"""),
    md(r"""
## Where to go next

Read [docs/08_programmable_aperture.md](../docs/08_programmable_aperture.md) for the derivations, the choice of $W$, and the full slide-29 analysis. The matching tab in the Streamlit explorer (`streamlit run app/streamlit_app.py`) lets you change the pattern and the noise level. Notebook 05 extends the light field to high dynamic range.
"""),
]

# =============================================================================
# 05  HDR light fields
# =============================================================================
NB05 = [
    md(r"""
# 05. HDR light fields
"""),
    md(r"""
A single exposure clips the highlights and buries the shadows in noise, and refocusing makes this worse: it averages many rays, so one clipped highlight leaks into every refocused photo. HDR light fields capture more range by varying exposure, either across the views within one shot or by bracketing the whole light field over time. This notebook builds the classic HDR tool chain (response calibration, merging, tone mapping, exposure fusion) on a synthetic light field with a dynamic range of several thousand to one. It compares one-shot per-view bracketing with full temporal bracketing and a single-exposure baseline, and ends by showing why refocusing must happen on linear radiance.
"""),
    code(HEADER + r"""
import lf
from lf import hdr, refocus, LightField

rad_lf = hdr.synthetic_hdr_light_field(n_views=7, size=128)
gt_c = rad_lf.center_view
print(rad_lf, '| measured dynamic range:', round(rad_lf.meta['dynamic_range']))
"""),
    md(r"""
## The scene

The views hold linear radiance, not values in $[0, 1]$, so they must be tone mapped to be displayed. The grass is dark, the text is mid-grey and the gravel disc is very bright, and a single 8-bit photo cannot show all three well.
"""),
    code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].imshow(hdr.tonemap(gt_c)); axes[0].set_title('ground-truth radiance (tone mapped)'); axes[0].axis('off')
im = axes[1].imshow(rad_lf.meta['gt_disparity'], cmap='viridis')
axes[1].set_title('ground-truth disparity'); axes[1].axis('off')
fig.colorbar(im, ax=axes[1], shrink=0.8)
plt.tight_layout()
"""),
    md(r"""
The ground truth needs a tone-mapping operator to be shown at all, which is the first hint that the display step is a separate stage from the measurement.
"""),
    md(r"""
## Three exposures of the centre view

We capture the centre view at three exposures with the gamma camera model from `simulate_capture`. The exposures are $1/32$, $1$ and $32$, which is the range that covers this scene; the brief's $1/8$ to $8$ would miss the bright gravel and the dark grass.
"""),
    code(r"""
exposures = (1 / 32, 1.0, 32.0)
rng = np.random.default_rng(0)
ldr_c = [hdr.simulate_capture(gt_c, t, crf='gamma', rng=rng) for t in exposures]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, t, img in zip(axes, exposures, ldr_c):
    ax.imshow(img); ax.set_title(f'exposure {t:g}'); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The $1/32$ shot keeps the gravel detail, but the grass is almost black and noisy. The $32$ shot shows the grass well, but the gravel and the text are clipped to white. No single shot holds the whole range.
"""),
    md(r"""
## Recovering the camera response

`solve_response` estimates the log inverse response $g$ by least squares, from pixels seen at several exposures. The scale of $g$ is fixed at the mid-grey level, so we compare the shape with the gamma-2.2 law that the simulator used.
"""),
    code(r"""
samples = hdr.sample_pixels(ldr_c, n=200, rng=np.random.default_rng(0))
g, ln_e = hdr.solve_response(samples, exposures)
z = np.arange(256) / 255.0
g_true = 2.2 * (np.log(np.maximum(z, 1e-3)) - np.log(128 / 255.0))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(np.arange(256), g, label='recovered g (Debevec and Malik)')
ax.plot(np.arange(256), g_true, '--', label='gamma 2.2 law (truth)')
ax.set_xlabel('pixel value Z'); ax.set_ylabel('ln(E dt)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
"""),
    md(r"""
The recovered curve follows the gamma law in the mid-range where the three exposures overlap. At the ends, the curve is dominated by noise near black and by clipping near white, so the hat weights should down-weight those levels.
"""),
    md(r"""
## Merging and tone mapping

`merge_hdr` combines the three exposures into a radiance image, using the hat weights to trust mid-range values. `tonemap` then makes it displayable. `exposure_fusion` is the Mertens alternative: it blends the display images directly, without a response curve.
"""),
    code(r"""
inv = hdr.inverse_crf('gamma')
rad_merge = hdr.merge_hdr(ldr_c, exposures, response='gamma')
fused = hdr.exposure_fusion(ldr_c)
print(f"{'estimate':28s} {'log-RMSE':>10s} {'PSNR tone-mapped':>18s}")
print(f"{'single exposure 1.0':28s} {hdr.log_rmse(inv(ldr_c[1]), gt_c):10.3f} {hdr.psnr_tonemapped(inv(ldr_c[1]), gt_c):18.2f}")
print(f"{'merged (3 exposures)':28s} {hdr.log_rmse(rad_merge, gt_c):10.3f} {hdr.psnr_tonemapped(rad_merge, gt_c):18.2f}")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
panels = [('single exposure 1.0', ldr_c[1]), ('exposure fusion (Mertens)', fused),
          ('merged, then tone mapped', hdr.tonemap(rad_merge)), ('ground truth', hdr.tonemap(gt_c))]
for ax, (title, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
The merged radiance keeps the gravel and the grass at once, and it is much closer to the ground truth than any single exposure. Exposure fusion gives a pleasing display image without radiance, but it cannot be refocused linearly, so the rest of the notebook uses radiance.
"""),
    md(r"""
## Strategy A: one exposure per view

Strategy A captures every view at its own exposure in one shot. We use a checkerboard exposure pattern, so each view sees all three exposures among its neighbours. `merge_bracketed_views` warps every view onto the target using the disparity map, merges them and rejects the ones that disagree.
"""),
    code(r"""
em = hdr.exposure_pattern(rad_lf.nu, rad_lf.nv, kind='checker')
print('exposure per view (rows = u, columns = v):')
print(em)
ldr_A, em = hdr.capture_bracketed_views(rad_lf, em, rng=np.random.default_rng(0))
hdr_A = hdr.merge_bracketed_views(ldr_A, em, disparity=rad_lf.meta['gt_disparity'])
per_view = hdr.evaluate_hdr_lf(hdr_A, rad_lf)
hdr_A_nodepth = hdr.merge_bracketed_views(ldr_A, em, disparity=None)
print('Strategy A, with ground-truth disparity: mean log-RMSE', round(per_view['mean'], 3),
      '| centre', round(per_view['center'], 3))
print('Strategy A, no depth (no warping)       : mean log-RMSE',
      round(hdr.evaluate_hdr_lf(hdr_A_nodepth, rad_lf)['mean'], 3))
"""),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
axes[0].imshow(hdr.tonemap(hdr_A.center_view)); axes[0].set_title('strategy A, centre view'); axes[0].axis('off')
im1 = axes[1].imshow(em, cmap='viridis'); axes[1].set_title('exposure per view (u, v)'); axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], shrink=0.8)
im2 = axes[2].imshow(per_view['per_view'], cmap='magma'); axes[2].set_title('log-RMSE per view'); axes[2].axis('off')
fig.colorbar(im2, ax=axes[2], shrink=0.8)
plt.tight_layout()
"""),
    md(r"""
One shot recovers the scene with a mean log-RMSE of 0.49 (0.30 at the centre view). Dropping the depth map raises the error to 1.04, so the depth-based warp is what makes one-shot capture work.
"""),
    md(r"""
## Strategy C: temporal bracketing

Strategy C captures the whole light field at all three exposures, three shots in total, and merges each view independently. It needs no depth, so it is an upper bound for strategy A.
"""),
    code(r"""
C = hdr.temporal_bracket_hdr_lf(rad_lf, exposures)
one_lf = LightField(inv(hdr.capture_bracketed_views(rad_lf, np.full((7, 7), 1.0),
                                                   rng=np.random.default_rng(0))[0].views), 'one_exposure')
rows = [('1 shot, one exposure (1.0)', 1, hdr.evaluate_hdr_lf(one_lf, rad_lf)),
        ('strategy A (1 shot, per view)', 1, per_view),
        ('strategy C (3 shots, temporal)', C['shots'], hdr.evaluate_hdr_lf(C['hdr_lf'], rad_lf))]
print(f"{'method':32s} {'shots':>5s} {'mean log-RMSE':>14s} {'centre log-RMSE':>16s}")
for name, shots, res in rows:
    print(f"{name:32s} {shots:5d} {res['mean']:14.3f} {res['center']:16.3f}")
"""),
    md(r"""
Strategy C is the most accurate, as expected, but it takes three times the shots. Strategy A has about twice its error (0.49 against 0.25) in one shot, and it is far better than a single exposure (1.30).
"""),
    md(r"""
## Refocusing: LDR versus radiance

The last step is refocusing. `ldr_refocus_comparison` refocuses at $\alpha = 1.5$ (the gravel) in three orders: average the display values of a single exposure, average linear values and tone map last, or tone map each view and then refocus. `hdr_refocus` is the correct order, shear and average linear radiance and tone map at the end.
"""),
    code(r"""
alpha_f = 1.5
cmp = hdr.ldr_refocus_comparison(rad_lf, alpha_f, exposure=1.0)
hdr_A_img, hdr_A_rad = hdr.hdr_refocus(hdr_A, alpha_f, return_radiance=True)
print(f"{'refocus order':40s} {'log-RMSE (radiance)':>20s} {'PSNR (display)':>15s}")
print(f"{'LDR: average display values':40s} {cmp['log_rmse_ldr']:20.3f} {cmp['psnr_display_ldr']:15.2f}")
print(f"{'LDR: linearise, then average':40s} {cmp['log_rmse_ldr_lin']:20.3f} {cmp['psnr_display_ldr_lin']:15.2f}")
print(f"{'tone map each view, then refocus':40s} {'-':>20s} {cmp['psnr_display_tm_first']:15.2f}")
print(f"{'radiance refocus (strategy A HDR)':40s} {hdr.log_rmse(hdr_A_rad, refocus.refocus(rad_lf, alpha_f)):20.3f} {'-':>15s}")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
panels = [('LDR: average display values', cmp['ldr']), ('tone map each view, then refocus', cmp['tm_first']),
          ('radiance refocus (strategy A)', hdr_A_img), ('ground truth', cmp['gt_display'])]
for ax, (title, img) in zip(axes, panels):
    ax.imshow(np.clip(img, 0, 1)); ax.set_title(title, fontsize=10); ax.axis('off')
plt.tight_layout()
"""),
    md(r"""
Averaging clipped display values is the least accurate (log-RMSE 1.15), and linearising first helps only a little (1.10), because clipping has already destroyed the gravel. Refocusing the radiance and tone mapping last (third panel) keeps the bright gravel, which the display-first orders clip, and that is why the rule is to average linear values and display last. The coloured fringes at the gravel edge in the third panel are residual error from the one-shot merge.
"""),
    md(r"""
## Depth-free refocusing

When no depth map is available, `merge_bracketed_refocused` merges in the refocused domain: it shears each exposure group by $\alpha$ and merges the groups as an exposure stack. It is exact only on the plane $\alpha$ itself, so the error here is higher than strategy A with depth.
"""),
    code(r"""
print(f"{'alpha':>6s} {'depth-free':>12s} {'with depth':>12s}   (log-RMSE vs ground-truth refocus)")
for a in (-1.0, 0.0, 1.5):
    ref_gt = refocus.refocus(rad_lf, a)
    ref_free = hdr.merge_bracketed_refocused(ldr_A, em, alpha=a)
    ref_depth = hdr.hdr_refocus(hdr_A, a, return_radiance=True)[1]
    print(f"{a:6.1f} {hdr.log_rmse(ref_free, ref_gt):12.3f} {hdr.log_rmse(ref_depth, ref_gt):12.3f}")
"""),
    md(r"""
The depth-free merge is exact only on the plane it is sheared to. At $\alpha = 1.5$ (the gravel) it beats the depth-based merge here, but at $\alpha = -1$ and $0$ its error is much larger, while the depth-based merge stays accurate everywhere. Depth-free merging is a shortcut for one plane, not a general refocus.
"""),
    md(r"""
## Where to go next

Read [docs/09_hdr_light_field.md](../docs/09_hdr_light_field.md) for the derivations, the four strategies, the aperture-area bracketing that this notebook leaves out, and the table of results. The matching tab in the Streamlit explorer (`streamlit run app/streamlit_app.py`) lets you change the exposure pattern and the scene gain.
"""),
]


if __name__ == "__main__":
    build("01_light_field_basics.ipynb", NB01)
    build("02_refocus_aperture_perspective.ipynb", NB02)
    build("03_depth_and_edof.ipynb", NB03)
    build("04_programmable_aperture.ipynb", NB04)
    build("05_hdr_light_fields.ipynb", NB05)
