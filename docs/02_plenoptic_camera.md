# 02. The plenoptic (light-field) camera

## 1. In one paragraph

A plenoptic camera is an ordinary camera with a micro-lens array placed where the film used to be, and the sensor moved one micro-lens focal length behind the array. Each micro-lens turns the bundle of rays reaching it into a tiny image of the main-lens aperture, so the pixels under one micro-lens record *where on the lens* each ray came from. The raw sensor image is therefore the 4-D light field `L(u, v, s, t)` written out as a mosaic of small tiles. Decoding it (taking the same pixel under every micro-lens) gives the sub-aperture images used by every other module in `lf/`. The price is resolution: spatial resolution is the sensor resolution divided by the angular resolution (Ng et al. 2005, lecture 15; lecture 16 slide 14).

## 2. Theory

**Two-plane parameterisation (slides 24-27).** A ray is described by where it crosses the lens plane, `(u, v)`, and the sensor plane, `(s, t)`. The 4-D "light field picture" of slide 25-27 is a 2-D array of tiles: tile `(s, t)` shows the `(u, v)` samples under micro-lens `(s, t)`; equivalently, each *sub-square at fixed* `(u, v)` is the image formed only by rays through that point of the lens (a pinhole view). Adelson & Wang (1992) described the same idea as "single lens stereo": a lens plus a lenticular array records the viewpoint-dependent structure behind one lens. Lippmann (1908) proposed the micro-lens "integral photography" idea; Levoy & Hanrahan (1996) gave the `(u,v,s,t)` formalism for rendering.

**Conventional vs light-field camera (slides 29-30).** A conventional sensor pixel at `(s, t)` integrates over the whole `(u, v)` aperture: `E(s,t) = sum_{u,v} L(u,v,s,t)`. The light-field camera keeps the sum un-done: it stores the integrand.

**Optical design (slides 31-37).**
* The main lens focuses the subject *onto the micro-lens array*.
* The micro-lenses are focused at the main lens (optical infinity for them), so the sensor sits one micro-lens focal length `f_m` behind the array. Each micro-lens then forms an image of the aperture of diameter `d = f_m / N_main` (`N_main` = main-lens f-number).
* Number of pixels behind each micro-lens = angular resolution; number of micro-lenses = spatial resolution.
* **f-number matching (slide 35).** The micro-lens f-number is `f_m / p` (`p` = pitch). If `N_main = f_m / p`, the aperture images exactly tile the sensor. With the Ng prototype (`f_m = 500 um`, `p = 125 um`) both are f/4. At f/2.8 the discs are 179 um wide and spill into neighbours (cross-talk); at f/8 they are 62 um wide and most pixels are unused.
* Prototype (slide 37): 4000 x 4000 sensor (Kodak 16 MP, 9 um pixels) behind about 292 x 292 micro-lenses: 14 x 14 pixels per lens (125 um / 9 um = 13.9). Medium-format Contax camera as main body.

`lf.plenoptic.design()` reproduces these numbers:

| main lens | disc diameter | overlap (cross-talk) | unlit part of cell |
|---|---|---|---|
| f/2.8 | 179 um | 38 % | 0 % |
| f/4 (matched) | 125 um | 0 % | 21 % |
| f/8 | 62 um | 0 % | 80 % |

Even when matched, a circular aperture image inscribed in a square cell leaves 1 - pi/4 = 21 % of pixels dark. (4000/14 = 285 and 4000 x 9 / 125 = 288 micro-lenses; the paper's 292 corresponds to a 123 um pitch. The difference is just rounding in the slide.)

**Resolution trade-off (lecture 16 slides 6, 14).** Spatial resolution = sensor pixels / angular samples. Total number of rays is about the sensor pixel count.

| sensor | angular | spatial (slide) | spatial (`resolution_tradeoff`) |
|---|---|---|---|
| 4000 x 4000 (Ng prototype; slide says "Lytro") | 14 x 14 | 292 x 292 | 285 x 285 |
| 2656 x 1216 | 16 x 16 | 166 x 76 | 166 x 76 |
| 4044 x 3240 | 5 x 4 | 808 x 810 | 808 x 810 |
| 1629 x 2052 | 9 x 9 | 181 x 228 | 181 x 228 |

## 3. Algorithm

*Simulating the raw image* (`to_lenslet`): for every micro-lens `(s, t)` write the `U x V` angular samples `L[:, :, s, t]` as a tile at pixel `(sU, tV)`.

*Decoding* (`from_lenslet`, slide 41): sub-aperture image `(u, v)` = pixel `(u, v)` taken under every micro-lens, i.e. `raw[u::U, v::V]`.

*Conventional photo / digital stopping down* (`conventional_photo`, slide 43): average all pixels of each tile; or only a central disc of radius `r` pixels. This equals `refocus(lf, 0, mask=disc)` and is the shallow-vs-deep depth of field control of slide 65: the aperture is chosen *after* the exposure.

## 4. Implementation notes

* `to_lenslet(lf, invert=False)` / `from_lenslet(raw, angular, invert=False)`: exact inverses (tested). Layout `raw[s*U+i, t*V+j] = L[u(i), v(j), s, t]`.
* Orientation: a real micro-lens images the aperture *inverted*, so pixel `(i, j)` sees aperture `(U-1-i, V-1-j)`; `invert=True` flips. Default `False` matches the Stanford ESLF convention (`u = i % U`).
* `lenslet_closeup`, `disc_mask` (same centre `(U//2, V//2)` as `LightField.center`), `design`, `resolution_tradeoff`, `camera_table`.
* `simulate_fnumber_mismatch(lf, N_main, N_micro=4)`: scales the angular patch by `r = N_micro / N_main` about the tile centre (nearest neighbour), masks it to a disc, writes it into the mosaic and averages overlapping contributions. A toy model of slide 35, not a ray trace.
* Reshape/transposes only, so everything is O(pixels).
* Overlap/unlit fractions in `design` are geometric estimates (disc vs square cell, 400 x 400 sampling).

## 5. Experiments

* `docs/figures/lenslet_raw_synthetic.png`: simulated raw image of the synthetic scene and a 6 x 6 micro-lens close-up on the edge of the (out-of-focus) gravel disc. Tiles differ because the disc has disparity 1.5; in-focus regions (text plane) give nearly identical tiles, as on slide 40.
* `docs/figures/subaperture_extraction.png`: the same pixel highlighted under every micro-lens; gathered together they form a patch of one sub-aperture view.
* `docs/figures/conventional_vs_stopdown.png`: all pixels (blurry gravel/grass), a central disc, a single pixel (everything sharp). Noise rises as fewer pixels are averaged.
* `docs/figures/fnumber_matching.png`: f/2.8, f/4, f/8 as on slide 35.
* Real data: `lytro_views_grid.png`, `lytro_epi.png`, `lytro_vignetting.png`, `lego_views_grid.png`. 
  The Lytro ESLF PNG is 5264 x 7574 x 4 (RGBA, uint8) = 14 x 376 by 14 x 541 pixels; the loader drops alpha. De-interleaving with `u = i % 14, v = j % 14` gives coherent photographs: adjacent views differ by a mean absolute difference of 0.005 against 0.018 between corner views, the EPI lines are straight and nearly vertical (disparity of a fraction of a pixel per step), and horizontal-only and vertical-only shift-and-add peak at the same alpha (correlation 0.97 over 32 textured tiles, alpha about -0.5 .. +0.5). The 14 x 14 grid is circular: the four 3-4-view corners are black, and the mean intensity falls from 0.10 at the centre to about 0.07 at the edges of the central 11 x 11. The central 9 x 9 (the default) keeps every view above 0.07 mean intensity and is a good compromise; 11 x 11 also works but is dimmer and noisier. The scene (blossoms against a dark sky) has no clearly nearest object, so the sign of alpha for near objects could not be confirmed from this image alone.

## 6. Try it

```python
from lf import io, plenoptic as P
lf = io.load('synthetic', n_views=9, size=128)
raw = P.to_lenslet(lf); back = P.from_lenslet(raw, (9, 9))
photo = P.conventional_photo(raw, (9, 9), radius=2)
print(P.design(main_f_number=2.8))
```
`python scripts/make_figures_plenoptic.py` regenerates all figures; `pytest tests/test_plenoptic.py`.

## 7. Limitations and extensions

* No microlens-grid calibration: real Lytro raw files are hexagonally packed, rotated, vignetted and Bayer-mosaiced; Dansereau et al. (2013) show how to calibrate, demosaic and resample them. The Stanford ESLF files are already decoded to a square 14 x 14 grid.
* Our mismatch simulation ignores diffraction, micro-lens aberrations and real overlap physics (sums, not averages).
* The ESLF relation: a Stanford ESLF PNG *is* the `to_lenslet` mosaic with `invert=False`, 14 x 14 pixels per tile, 376 x 541 tiles. Outer views are dark and noisy (vignetting), hence `crop_angular=9` by default in `lf.io.load`.
* Extensions: hexagonal grids, Bayer demosaicing before decoding, estimating the vignetting profile and flat-fielding the views.

## 8. References

* R. Ng, M. Levoy, M. Bredif, G. Duval, M. Horowitz, P. Hanrahan, "Light field photography with a hand-held plenoptic camera", Stanford Tech Report CTSR 2005-02.
* E. H. Adelson, J. Y. A. Wang, "Single lens stereo with a plenoptic camera", IEEE TPAMI 14(2), 1992.
* G. Lippmann, "Epreuves reversibles donnant la sensation du relief", J. Phys. Theor. Appl. 7(1), 1908.
* M. Levoy, P. Hanrahan, "Light field rendering", SIGGRAPH 1996.
* A. S. Raj, M. Lowney, R. Shah, G. Wetzstein, "Stanford Lytro Light Field Archive", 2016, http://lightfields.stanford.edu/LF2016.html.
* D. G. Dansereau, O. Pizarro, S. B. Williams, "Decoding, calibration and rectification for lenselet-based plenoptic cameras", CVPR 2013.
