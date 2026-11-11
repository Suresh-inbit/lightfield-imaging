# 09 - HDR light fields

Code: `lf/hdr.py`, tests: `tests/test_hdr.py`, figures: `scripts/make_figures_hdr.py`.

## 1. In one paragraph

A sensor clips bright regions and drowns dark ones in noise, and a light field makes this worse in an unusual way: refocusing *averages* many rays, so one clipped highlight biases every refocused image, and the bokeh of a bright out-of-focus lamp is the first thing to go wrong. HDR light fields attack this by varying exposure while capturing: either over time (bracket the whole light field, 3x the shots), or across the angular samples themselves (each sub-aperture view gets its own exposure, one shot), or across aperture area (a programmable aperture, lecture 16). This note builds the classic HDR toolbox (Debevec-Malik, Reinhard, Mertens), a synthetic ~5000:1 light field, and compares four strategies by log-RMSE against known radiance. The headline result: averaging has to happen on linear radiance, and per-view bracketing with depth recovers most of the quality of full bracketing at a third of the shots.

## 2. Theory

**Camera model.** A pixel reports `Z = f(E dt + n)`, clipped to `[0, 1]` and quantised, where `E` is scene radiance, `dt` the exposure (shutter time, gain, or open aperture area), `f` the camera response curve (CRF), `n` photon (Poisson) plus read noise. Saturation caps `E dt` at the full well; darkness leaves `E dt` below the noise floor. Only a middle band of `E` is informative at any one exposure, and exposures shift that band.

**Debevec & Malik (1997).** Define `g = ln f^-1`, so `g(Z) = ln E + ln dt`. For `N` pixels seen at `P` exposures, `solve_response` minimises
`sum_ij w(Z_ij) [g(Z_ij) - ln E_i - ln dt_j]^2 + lambda sum_z w(z) g''(z)^2`
with `g(128)=0` to fix the scale; the second term is a smoothness prior filling levels no sample hit. `w` is the hat weight (`weight_hat`), which trusts mid-range values. Once `g` is known, `merge_hdr` computes `ln E = sum_j w(Z_j)[g(Z_j) - ln dt_j] / sum_j w(Z_j)`. Pixels clipped in every exposure get zero weight; we then fall back to the shortest exposure (a lower bound) instead of NaN, and symmetrically to the longest for all-dark pixels.

**Reinhard et al. (2002)** global tonemapping: scale luminance by `key / Lbar` (`Lbar` the log-average), then `Ld = L(1 + L/white^2)/(1 + L)`; colours keep their ratios. `tonemap` also offers `log` and `gamma`.

**Mertens et al. (2007)** exposure fusion skips radiance altogether. Per pixel quality `W = C^a S^b E^c` (contrast, saturation, well-exposedness), normalised over the stack, blended in a Laplacian pyramid so that seams between exposures vanish. It gives a pleasant LDR picture but not radiance, so it cannot be fed to refocusing.

**Light fields and exposure.** Each sub-aperture view is a pinhole-like image, so any of the above applies per view. Two additional facts matter. (i) Views are nearly redundant: with the disparity `d` known, view `(u',v')` predicts view `(u,v)` as `L(u',v', s + d(u'-u), t + d(v'-v))`, so exposures can be *spread* over the angular grid and pooled. (ii) The refocused image is `mean_{u,v} E(...)` of *linear radiance* (a photon sum); gamma-encoded or clipped values break that identity.

Lecture link: lecture 16 slides 18-19 list HDR imaging (Debevec & Malik) under multiple exposures and Nayar & Branzoi under coded aperture. The programmable aperture (Liang et al. 2008) opens a chosen set of cells per shot, so it controls exposure as well as angular content. Plenoptic cameras (Ng 2005) vignette, so rim views are dimmer: an uncontrolled version of strategy A.

## 3. Algorithm

**A. one exposure per view (1 shot).** (1) pick an exposure map over `(u,v)` (`exposure_pattern`); (2) capture each view with its exposure (`capture_bracketed_views`); (3) for each target view, re-project the centre disparity to it (z-buffered splat), warp all other views onto the target with `map_coordinates`, linearise, hat-weight and merge in the log domain (`merge_bracketed_views`); (4) one robust re-weighting pass drops sources whose ln-radiance disagrees with the estimate (occlusion ghosts). Depth-free variant `merge_bracketed_refocused(alpha)`: per exposure group, shear by `alpha` and average, then merge the groups as an exposure stack.

**B. refocus in radiance.** `hdr_refocus`: shear-and-average linear radiance, tonemap last. `ldr_refocus_comparison` contrasts it with clipping first and tonemapping first.

**C. temporal bracket (3 shots).** Capture the whole light field at each exposure; per-view `merge_hdr`. No depth needed; the quality reference.

**D. aperture-area bracketing (3 shots).** Open 1, 3, 9 cells at fixed shutter; image = sum of the sheared open views, exposure `proportional to area`; merge as an exposure stack (`aperture_area_bracketing`).

## 4. Implementation notes

* HDR light fields store **linear radiance** in `views`, float32, not clipped: an explicit exception to the `[0,1]` convention, flagged by `meta['radiance']=True`. `meta['gt_disparity']` and `meta['dynamic_range']` are kept.
* `response` can be `None` (linear), a name (`'gamma'`, `'srgb'`), a callable inverse CRF, or the `g` array from `solve_response` (which is defined only up to a global scale: align before comparing absolute values).
* The camera model `simulate_capture` does scale, Poisson + Gaussian noise, clip, CRF, quantise, in that order (clipping before the CRF).
* Warping at target view `(u,v)` needs the disparity *of that view*. Backward fixed-point iteration cannot find foreground solutions across a depth edge (it silently picks the background), which put a crescent of error around the disc; a forward z-buffered splat of the centre map (nearest wins) fixes it, with the fixed point only used to fill disocclusion holes.
* Cost of merge A: `(U V)^2` warps, about 15 s for 7x7 views at 128x128.
* Default exposures are `(1/32, 1, 32)`, not the brief's `(1/8, 1, 8)`: the scene spans ~5000:1 and 64:1 cannot cover it. Gains `(0.03, 1, 40)`.
* The unequal noise of different exposures is not modelled by hat weights (they live in the gamma domain); inverse-variance weights (Granados et al. 2010) would be better, left as an extension.

## 5. Experiments

Setup: 7x7 views, 128x128, noise `photon_scale=2000`, read noise 0.005, 8 bit, gamma 2.2; log-RMSE is natural-log RMS error against the true radiance (0.69 = factor 2).

* `hdr_synthetic_scene.png`: the centre view at three exposures plus tonemapped ground truth. Each layer (dark grass, text, bright gravel) is only readable in one exposure.
* `hdr_response_curve.png`: recovered `g(Z)` against `2.2 ln(Z/255)`; correlation 0.988 on levels 9-249 (5 exposures, 200 samples, lambda 50).
* `hdr_exposure_patterns.png`: checker, rows, radial, random maps over the angular grid.
* `hdr_exposure_fusion.png`: Mertens fusion vs Debevec-Malik + Reinhard from the same three exposures; fusion looks good but its pixel values are not radiance.
* `hdr_strategyA.png`: bracketed LDR views, merged centre view with depth (log-RMSE 0.30) and without (0.81); without depth, the bright disc gets a coloured halo, because views are merged as if all points had `d = 0`.
* `hdr_refocus_ldr_vs_hdr.png`: three focus settings. Clip-then-refocus (col 1) has a hard-edged, flat-topped highlight with a dim rim; radiance refocus (col 3) matches the ground truth bokeh. Tonemap-then-refocus (col 2) shows a tonemapped-looking softer blob with the wrong brightness profile.
* `hdr_strategy_table.png` / `hdr_strategy_table.md`: table below.
* `hdr_lego_pseudo.png`: Lego Knights with pseudo-HDR (inverse gamma, top 2% boosted x16): the LDR and pseudo-radiance centre views, refocus in the display domain versus in radiance then tonemapped, and a difference map that is brightest on the wall and floor. Illustrative only; no real clipping information exists in an LDR light field.

Results (log-RMSE, lower is better):

| method | shots | LF log-RMSE | refocus a=0 | refocus a=1.5 |
|---|---|---|---|---|
| A checker, gt depth | 1 | 0.487 | 0.193 | 0.696 |
| A rows, gt depth | 1 | 0.613 | 0.226 | 0.880 |
| A radial, gt depth | 1 | 0.816 | 0.364 | 1.058 |
| A random, gt depth | 1 | 0.489 | 0.205 | 0.701 |
| A checker, no depth | 1 | 1.041 | 1.376 | 0.580 |
| A checker, refocused-domain merge | 1 | - | 0.786 | 0.367 |
| A checker, estimated depth (MAE 1.01 px) | 1 | 0.428 | 0.469 | 0.249 |
| C temporal bracket | 3 | 0.248 | 0.155 | 0.151 |
| single exposure x1/32 | 1 | 2.985 | 1.348 | 1.330 |
| single exposure x1 | 1 | 1.304 | 1.274 | 1.101 |
| single exposure x32 | 1 | 2.687 | 2.954 | 2.763 |
| D aperture area 1/3/9 | 3 | - | 1.408 | 0.914 |
| B refocus LDR, display average | 1 | - | 1.434 | 1.148 |
| B refocus LDR, linearised average | 1 | - | 1.274 | 1.101 |
| B refocus HDR from C, in radiance | 3 | - | 0.155 | 0.151 |

Observations. (1) Any single exposure is poor (>=1.3) because one of the three layers is always clipped or buried. (2) C is the upper bound (0.25); A with checker and gt depth reaches 0.49 with a third of the shots, and the errors are concentrated at depth edges (see the error map) where occlusion invalidates the warp. (3) Pattern matters: checker and random (every neighbourhood sees all exposures) beat rows and radial; radial is worst because the long-exposure views are all in the centre, so the rim never sees the dark layer well. This is the vignetting scenario: uncontrolled exposure variation helps but is far from the best pattern. (4) No depth is much worse for the LF (1.04) but is *not* uniformly bad: in the refocused domain at the matching alpha (1.5) the merge is better than with warped views (0.37), and its error is huge at alpha = 0 (0.79), where different groups have different bokeh. (5) The estimated-depth row (MAE 1 px, from a crude bootstrapped pipeline: log-linearised views fed to `lf.depth.depth_combined`) scored as well as or better than gt depth on the refocused metrics. We do not read much into this: the gt map has sharp edges that the splat/warp handles imperfectly, the smoothed estimate does not; it shows that the exact depth matters less than getting the *ordering* right at edges, but with one scene and one seed it is an observation, not a result. (6) D (aperture area) does not beat a single exposure by much: its range is only 9:1 and the three images have different depth of field.

**Linear-domain rule (B).** On the refocused image: clip-then-refocus 1.43 at alpha 0 vs 0.155 for radiance refocus (the HDR light field here is from C, so it includes capture noise); linearising the average before refocusing fixes the gamma but not clipping (1.27). Tonemapping every view before averaging gives 17.1 dB PSNR at alpha 0 against the tonemapped true refocus, because the tone curve is concave and the average of tonemapped values is not the tonemap of the average. Rule: capture, merge, refocus (averaging), then tonemap.

## 6. Try it

```python
from lf import hdr as H
gt = H.synthetic_hdr_light_field(7, 128)
a = H.strategy_a(gt, "checker")                 # dict: hdr_lf, ldr_lf, exposure_map, shots, log_rmse
c = H.temporal_bracket_hdr_lf(gt, (1/32, 1, 32))
img = H.hdr_refocus(a["hdr_lf"], alpha=1.5)     # display image
cmp = H.ldr_refocus_comparison(gt, 0.0, 1.0)    # LDR vs radiance refocus
```
`python scripts/make_figures_hdr.py` reproduces all figures and the table (about a minute); `python -m pytest tests/test_hdr.py -q`.

Which to choose: if you can afford 3x shots and the scene is static, use C (no depth, best quality). If you need one shot, a plenoptic or programmable aperture with a checker/random exposure pattern plus a depth estimate (A) works, but expect ghosting at depth edges. Use D only when the dynamic range is small (< ~10:1) and you need refocused photos only. Always refocus on linear radiance.

## 7. Limitations and extensions

* Assumptions: static scene, known (or estimated) CRF, known disparity for A. Saturated regions have no texture, so correspondence-based depth estimation fails exactly where HDR matters; depth from unsaturated views only is a chicken-and-egg problem.
* The merge assumes Lambertian surfaces; specular highlights, the very case that motivates HDR, move across views.
* Noise weighting is hat-based, not variance-based.
* Ideas: joint depth + HDR estimation (alternate: depth from exposure-normalised views, merge, re-estimate); co-design exposure and angular sampling using the multiplexing matrix of `lf.multiplex` with varying row weights (each row a mask with weights acting as an exposure); optimise the exposure pattern for a given scene histogram; learned HDR view synthesis from alternating-exposure views (e.g. Kalantari & Ramamoorthi, "Deep high dynamic range imaging of dynamic scenes", SIGGRAPH 2017, treats the related problem of merging misaligned exposures); spatially varying pixel exposures (Nayar & Mitsunaga 2000) combined with the angular axis.
* Pseudo-HDR from LDR data (`hdr_from_ldr_light_field`) is a demo; the clipped information is gone.

## 8. References

* P. Debevec, J. Malik. Recovering high dynamic range radiance maps from photographs. SIGGRAPH 1997.
* T. Mertens, J. Kautz, F. Van Reeth. Exposure fusion. Pacific Graphics 2007.
* E. Reinhard, M. Stark, P. Shirley, J. Ferwerda. Photographic tone reproduction for digital images. SIGGRAPH 2002.
* S. Nayar, T. Mitsunaga. High dynamic range imaging: spatially varying pixel exposures. CVPR 2000.
* S. Nayar, V. Branzoi. Adaptive dynamic range imaging: optical control of pixel exposures over space and time. ICCV 2003.
* R. Ng, M. Levoy, M. Bredif, G. Duval, M. Horowitz, P. Hanrahan. Light field photography with a hand-held plenoptic camera. Stanford TR 2005-02.
* C.-K. Liang, T.-H. Lin, B.-Y. Wong, C. Liu, H. Chen. Programmable aperture photography: multiplexed light field acquisition. SIGGRAPH 2008.
* M. Granados, B. Ajdin, M. Wand, C. Theobalt, H.-P. Seidel, H. Lensch. Optimal HDR reconstruction with linear digital cameras. CVPR 2010.
* N. K. Kalantari, R. Ramamoorthi. Deep high dynamic range imaging of dynamic scenes. SIGGRAPH 2017.
