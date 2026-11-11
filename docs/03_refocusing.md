# 03 - Digital refocusing by shift-and-add

## 1. In one paragraph
A light field lets you choose the focal plane *after* the picture is taken. For every sub-aperture image you shift it by an amount proportional to its angular offset from the centre, then average. Objects whose disparity matches the shift line up and stay sharp; everything else is averaged over different positions and blurs, exactly like defocus in a real lens (Ng et al. 2005, lecture 15 slides 45-56).

## 2. Theory
**Two-plane parameterisation.** `L(u,v,s,t)`: a ray crosses the aperture plane at `(u,v)` and the sensor plane at `(s,t)`. A conventional photograph is `E(s,t) = sum_{u,v} L(u,v,s,t)` (slide 43: the sum under one microlens).

**Shift-and-add.** A scene point at depth with disparity `d` appears at `(s + d(u-u0), t + d(v-v0))` in view `(u,v)`. Sampling every view at `s + alpha (u-u0)` therefore gives
`I_alpha(s,t) = (1/N) sum_{u,v} L(u,v, s + alpha(u-u0), t + alpha(v-v0))`.
For `alpha = d` the term is constant in `(u,v)` for that point: it adds coherently (sharp). For other depths the samples come from `|d - alpha| * (u-u0)` pixels apart and blur; the blur diameter is `|d - alpha| * (U-1)` pixels, a linear function of the depth error. This is the "refocusing = shearing the light field then integrating over angle" picture.

**Windows under microlenses = shifted sub-aperture images.** The raw sensor under microlens `(s,t)` holds `L(., ., s, t)`. Summing the pixels `(u,v)` of the microlens at `(s + alpha(u-u0), ...)` (slides 45-50 and 54-55: windows extracted from several neighbouring microlenses) collects, for each `(u,v)`, one pixel from the sub-aperture image `u,v` at the shifted position. Both are the same sum over the same samples, re-indexed. The windows are shifted, so `alpha = 0` is the microlens plane.

**Link to Ng's alpha'.** Ng moves the film from depth `F` to `F' = alpha' F`. A ray hitting the new film at `s` and the lens at `u` hits the old film at `s_old = u + (s-u)/alpha' = s/alpha' + (1 - 1/alpha') u`, so
`E_{alpha'F}(s) ~ sum_u L(u, s/alpha' + (1-1/alpha') u)`.
Up to a global magnification `1/alpha'` this is a shift of each sub-aperture image by `(1-1/alpha') u`. With `p` pixels of image shift per angular step, our slope is `alpha = (1 - 1/alpha') p` (`lf.refocus.ng_alpha_to_disparity`, inverse `disparity_to_ng_alpha`). `alpha' = 1` -> 0.

**Digital refocusing theorem (slide 56).** An f/N light-field camera with PxP pixels per microlens can synthesise images as sharp as an f/(N P) conventional camera, or shallow-DoF f/N images focused anywhere inside the f/(N P) depth of field. See doc 04.

## 3. Algorithm
1. For each view `(u,v)` compute shift `(-alpha(u-u0), -alpha(v-v0))`.
2. Shift the view by that vector (interpolating).
3. Weighted-average all views (weights = aperture mask, default uniform).

## 4. Implementation notes
* `lf.refocus.refocus(lf, alpha, method, mask)`; `method` in `'subpixel'` (cv2 bilinear, replicated borders), `'integer'` (per-view rounding) and `'fourier'` (phase ramp `exp(-2 pi i (f_s dy + f_t dx))`, summed in the frequency domain, one inverse FFT). `focal_stack`, `shear` (returns the whole sheared 5-D array) unchanged.
* Boundaries: bilinear/integer replicate the edge, so borders look smeared; Fourier is periodic, so content wraps around. Keep alphas modest or crop `|alpha| (U-1)/2` pixels.
* Interpolation: bilinear interpolation acts as a low-pass filter whose strength depends on the fractional part of each shift, so fractional alphas are slightly softer than integer ones (an alpha-dependent sharpness bias when comparing alphas with `sharpness`). Fourier shifts are exact for band-limited images but ring near hard edges (Gibbs) and wrap. On the synthetic scene at alpha=0.5 the two agree to ~56 dB PSNR in the interior.
* Speed (256x256, 9x9 views): subpixel 0.04 s, integer 0.04 s, fourier 0.27 s, `shear` (scipy) 0.7 s.
* `sharpness(img, mask)` = variance of the Laplacian; `sweep_sharpness(lf, alphas, roi)` finds the alpha where a region is in focus.
* `SUGGESTED_ALPHA_RANGE` holds usable ranges per dataset.

## 5. Experiments
* `docs/figures/refocus_sweep_synthetic.png` - five alphas on the 3-layer scene. Look for grass sharp at -1, text at 0, gravel at +1.5; other layers go smooth with a blur proportional to depth difference. The darkened border is the replicate-boundary effect.
* `docs/figures/focal_stack_montage.png` - eight slices of the same stack; each layer enters and leaves focus in turn.
* `docs/figures/refocus_sweep.gif` - the sweep animated.
* Tests (`tests/test_refocus.py`) check, for all three methods, that sharpness inside each layer peaks within 0.25 of the true alpha (text 0, gravel +1.5, grass -1.0), mask `None` == all-ones, and Ng round-trip.
* **Real data.** Lego Knights and Bracelet are loaded with `downsample=4, angular_crop=9`; the Lytro flowers at full resolution with the central 9x9 views. Measured ranges (`SUGGESTED_ALPHA_RANGE`, from `sweep_sharpness`): lego (-2.0, 1.5), bracelet (-1.0, 2.0), lytro (-1.5, 1.0). Disparities scale with `1/downsample`, so these are comparable only at the same downsample.
* `docs/figures/refocus_sweep_lego.png` (animated in `refocus_sweep_lego.gif`) - five alphas from -2.0 to +1.5. The castle wall is sharp near -1, the knights near 0 and the near floor at positive alpha; watch the focal plane travel from wall to floor.
* `docs/figures/refocus_sweep_bracelet.png` (animated in `refocus_sweep_bracelet.gif`) - alphas from -1.0 to +2.0. The far clasp and upper chain come into focus near 0, the near chain at the bottom near +1.0 to +1.5; both ends of the range are blurred.
* There is no Lytro sweep figure. The blossoms are sharpest between -0.5 and +0.3 on the full-resolution data.

## 6. Try it
```python
import lf
from lf import refocus as R
L = lf.synthetic.synthetic_light_field(n_views=9, size=256)
img = R.refocus(L, 1.5, method='fourier')
import numpy as np; a = np.linspace(-2, 2.5, 19)
print(a[np.argmax(R.sweep_sharpness(L, a, roi=(slice(60,72), slice(58,70))))])
```
`python scripts/make_figures_refocus.py` regenerates all figures.

## 7. Limitations and extensions
* Few views (e.g. 7x7) and large |alpha| give *angular aliasing*: the out-of-focus blur is not a smooth disc but a grid of ghost copies, spaced `|alpha - d|` pixels apart (see the faint lattice on the gravel at alpha = 0). More views, or interpolating the angular dimension, fixes it.
* Occlusion is not modelled: foreground edges leak into the refocused background.
* Extension: shift-and-add with depth-aware weights (see `lf.depth`, `lf.edof`), Fourier-slice refocusing in O(N^2 log N) per image (Ng 2005 "Fourier slice photography").

## 8. References
* R. Ng, M. Levoy, M. Bredif, G. Duval, M. Horowitz, P. Hanrahan, "Light field photography with a hand-held plenoptic camera", Stanford Tech Report CTSR 2005-02; SIGGRAPH 2005 (Fourier slice paper: R. Ng, "Fourier slice photography", ACM TOG 24(3), 2005 - mentioned only).
* M. Levoy, P. Hanrahan, "Light field rendering", SIGGRAPH 1996.
* A. Isaksen, L. McMillan, S. Gortler, "Dynamically reparameterized light fields", SIGGRAPH 2000.
