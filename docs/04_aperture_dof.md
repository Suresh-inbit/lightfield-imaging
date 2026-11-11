# 04 - Aperture, depth of field and noise

## 1. In one paragraph
Averaging only some of the sub-aperture views is a *digital aperture*: using a small central disc is stopping down the lens after the fact (slide 43). A smaller aperture gives a larger depth of field (DoF) but collects less light, so the image is noisier; the digital refocusing theorem (slide 56) says how far you can push this.

## 2. Theory
A defocused point of disparity error `e = d - alpha` blurs over a patch of diameter `|e| * D`, where `D` is the diameter (in views) of the aperture region used. Halving `D` halves the blur, hence the DoF doubles (the same as doubling the f-number). The light collected is proportional to the number of views `n`; with independent per-view noise of std `sigma` the average has noise `sigma/sqrt(n)`.

**Digital refocusing theorem (slide 56).** For an f/N main lens and PxP pixels per microlens, a single pixel per microlens (window width 1) gives an image with the DoF of an f/(N P) camera; the full window gives f/N; a window of width `w` gives f/(N P / w). Refocusing then lets you place the (small) DoF of the f/N image anywhere inside the f/(N P) DoF. `effective_f_number(f_main, P, radius_in_pixels)` implements `N P / w` with `w = 2r+1` clipped to `[1,P]`.

## 3. Algorithm
1. Build an aperture mask `m(u,v)` (circle, square, annulus; hard or soft edge).
2. Refocus at the chosen alpha with that mask (normalised weighted average).

## 4. Implementation notes
* `aperture_mask(nu, nv, shape, radius, center, soft)`, `stop_down(lf, radius, alpha, shape, center)`.
* Radius is in angular steps. Radius 0 selects the central view only. Soft masks use a one-step-wide linear ramp.
* A non-centred mask or an annulus gives a different bokeh (annulus -> ring-shaped blur) and, off-centre, a view-shifted image.

## 5. Experiments
* `docs/figures/stopdown_comparison.png` - top row: noise-free refocus at alpha=0 with radius 1, 2, full; the number is the Laplacian-variance sharpness in the (out-of-focus) grass region. It drops monotonically (2.9e-3, 7.4e-4, 4.3e-5): more aperture = shallower DoF. Bottom row: the same with Gaussian noise (sigma=0.05) added to every view: the full aperture is visibly cleaner, the radius-1 image grainier (5 views, ~sqrt(81/5) = 4x more noise std). This is the DoF/SNR trade-off of a real lens. Note the radius-1 sharpness number would also include noise if measured on the noisy image, so it is measured on the clean one.
* The text layer at alpha=0 stays sharp in all panels (it is the focal plane).
* `stopdown_comparison.png` is synthetic only; no real-data stop-down figure is generated.
* `tests/test_aperture.py` verifies `stop_down(radius=0)` reproduces the centre view and a huge radius equals `refocus`.

## 6. Try it
```python
from lf import aperture as A, synthetic
L = synthetic.synthetic_light_field(9, 160)
img = A.stop_down(L, radius=1, alpha=0.0)
print(A.effective_f_number(4.0, 9, 1))   # f/12
```

## 7. Limitations and extensions
* The digital aperture is only as fine as the angular sampling: with 9x9 views you get a handful of discrete f-numbers.
* Real plenoptic cameras have vignetting and cross-talk between microlenses (outer views are darker/noisier), not modelled.
* Try a circular vs. annulus vs. square mask and look at the blur shape (bokeh).

## 8. References
* Ng et al., "Light field photography with a hand-held plenoptic camera", Stanford TR 2005-02 / SIGGRAPH 2005.
* Ng, "Fourier slice photography", SIGGRAPH 2005 (mention only).
