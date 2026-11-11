# Conventions (read this first)

This file defines the data conventions, API shapes and documentation template
used by every module in `lf/`.  It is written for a research student who wants
to extend the code, and it is the contract the implementation agents followed.

## 1. Data convention

* A light field is `lf.LightField` with `views` of shape **(U, V, H, W, C)**,
  `float32` in `[0, 1]`.  `views[u, v]` is the sub-aperture image at angular
  index `(u, v)`; `u` indexes angular rows, `v` angular columns, and
  `(u0, v0) = lf.center`.  The *direction* of the angular axes is fixed by the
  disparity convention below (nearer = positive), not by the physical camera
  motion: for the Stanford gantry this means both `u` and `v` run opposite to
  the gantry's y and x axes (the loader flips the axes so this holds for every
  dataset; verified on Lego Knights, see `docs/01_datasets.md`).
* Two-plane parameterisation `L(u, v, s, t)` (Ng 2005): `(u, v)` on the
  aperture plane, `(s, t)` on the sensor plane.  In arrays `s` = image row,
  `t` = image column.
* **Disparity** `d` of a scene point = shift in pixels per unit angular step.
  A point with disparity `d` that sits at `(s, t)` in the central view appears
  at `(s + d (u - u0), t + d (v - v0))` in view `(u, v)`.  Points on the
  rectification plane have `d = 0`; nearer points have `d > 0`.
* **Refocusing on the plane of disparity α** is therefore
  `I_α(s, t) = (1/N) Σ_{u,v} L(u, v, s + α (u - u0), t + α (v - v0))`,
  i.e. shift view `(u, v)` by `(-α (u - u0), -α (v - v0))` and average.
  On the synthetic scene (`lf.synthetic.synthetic_light_field`) the text plane
  is sharp at `α = 0`, the gravel disc at `α = 1.5`, the grass at `α = -1`.
  Ng's refocus parameter `α_Ng = F'/F` relates to our slope by
  `α = (1 - 1/α_Ng) · (pixel pitch of a unit aperture step)`; see `docs/03_refocusing.md`.
* Ground truth disparity for synthetic data: `lf.meta['gt_disparity']` (H, W).

## 2. Code conventions

* NumPy + SciPy (+ OpenCV / scikit-image where handy).  No new dependencies.
* Pure functions, `float32` in / out, **no plotting inside library functions**
  (plotting lives in `scripts/make_figures_*.py`, the notebooks and the app).
* Every public function has a docstring with: one-line summary, the maths
  (plain text / LaTeX-ish), parameters, returns, and a *Reference* line.
* Keep memory in check: the Lego Knights LF at full resolution is 3.6 GB as
  float32; load it with `lf.io.load('lego', downsample=4, angular_crop=9)`.
* Tests: `tests/test_<module>.py`, run with `python -m pytest tests -q`.
  Use the synthetic LF (`n_views=7, size=96`) so each test file finishes in
  well under a minute.  Tests must be deterministic (`seed=0`).
* Figures: `scripts/make_figures_<module>.py` writes PNGs to `docs/figures/`
  and must run headless (`matplotlib.use('Agg')`).  Use real data when
  `lf.io.available()[name]` is `True`, otherwise synthetic.

## 3. Documentation template (`docs/NN_<topic>.md`)

Each method document uses the same skeleton so a reader can navigate quickly:

1. **In one paragraph** — what the method does and why it matters.
2. **Theory** — the model and the derivation, with equations.  Connect to the
   lecture slides (EE 5176 lectures 15 & 16) and the original paper.
3. **Algorithm** — numbered steps as implemented.
4. **Implementation notes** — which functions, key design choices, complexity,
   numerical pitfalls (interpolation, boundaries, sign conventions).
5. **Experiments** — figures (from `docs/figures/`), what to look for, and the
   quantitative results on the synthetic scene (where ground truth exists).
6. **Try it** — how to reproduce interactively (app tab / notebook / one-liner).
7. **Limitations and extensions** — what breaks, what a student could try next.
8. **References** — full citations.

Write like a good tutorial: short paragraphs, equations only where they carry
information, every figure explained.

## 4. Public API (what the app and notebooks rely on)

```
lf.refocus.refocus(lf, alpha, method='subpixel', mask=None)       -> (H, W, C)
lf.refocus.focal_stack(lf, alphas, **kw)                           -> (n, H, W, C)
lf.aperture.aperture_mask(nu, nv, shape='circle', radius=None, center=None) -> (U, V)
lf.aperture.stop_down(lf, radius, alpha=0.0, shape='circle', center=None) -> (H, W, C)
lf.aperture.synthesize_view(lf, u, v)                              -> (H, W, C)  # fractional (u, v)
lf.depth.depth_from_focus(lf, alphas, ...)                         -> dict(disparity, confidence, ...)
lf.depth.depth_from_correspondence(lf, alphas, ...)                -> dict(disparity, confidence, ...)
lf.depth.depth_combined(lf, alphas, ...)                           -> dict(disparity, ...)
lf.depth.evaluate(disparity, gt, mask=None)                        -> dict(mae, rmse, bad_pixel_ratio)
lf.edof.all_in_focus(lf, alphas, ...)                              -> dict(image, index_map, ...)
lf.plenoptic.to_lenslet(lf)                                        -> (U*H, V*W, C) raw sensor mosaic
lf.plenoptic.from_lenslet(raw, angular)                            -> LightField
lf.plenoptic.conventional_photo(lf or raw, ...)                    -> (H, W, C)
lf.multiplex.s_matrix(n) / hadamard patterns, capture(), demultiplex(), snr analysis
lf.hdr.solve_response(), merge_hdr(), tonemap(), exposure_fusion(), light-field HDR pipelines
```
Modules may add more; the names above must exist with these meanings.
