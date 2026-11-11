# 08. Programmable aperture and multiplexed light-field acquisition

Code: `lf/multiplex.py`, `tests/test_multiplex.py`, `scripts/make_figures_multiplex.py`.
Lecture 16 slides 21-42 (Liang et al. 2008).

## 1. In one paragraph

A programmable aperture opens one region of the lens aperture per exposure.
Each exposure records, at every sensor pixel, one angular sample of the 4-D
light field, so after N exposures we hold an N-view light field at the **full
sensor resolution**; a plenoptic camera instead trades spatial for angular
resolution (Lytro: 16 MP becomes 14x14 views of 292x292, slides 6 and 14).
The price is noise: with a fixed total exposure time T each shot gets only T/N.
Multiplexing opens about half of the cells in every shot, collects far more
light per measurement, and recovers the views by inverting a known 0/1
matrix.  This document derives how much that helps (it depends strongly on the
noise type), compares with the plenoptic camera (slide 29), and reports where
the slide's numbers hold and where they need an extra assumption.

## 2. Theory

### 2.1 Sequential capture and exposure splitting (slides 21-24)

Let the aperture have N cells, cell j with radiance image `x_j` (our
`lf.views[u, v]`, `j = u V + v`).  Opening only cell j for time T/N gives, per
pixel, signal `(T/N) x_j`.  Every view has full spatial resolution, but each is
photon-starved: shot noise grows relative to signal, and the fixed read noise
`sigma_r` is added N times more strongly relative to a T-second exposure.

### 2.2 Multiplexing (slides 25-28)

Take K = N shots of duration `tau = T/K`; shot k opens the cells with
`W[k, j] = 1`:

    y_k = tau * sum_j W[k,j] x_j + n_k ,       y = tau W x + n ,
    x_hat = W^+ y / tau   (= W^-1 y / tau when W is square and invertible).

Noise model (as in `capture`): Poisson photon noise with `photon_scale` photons
per unit signal, so `var = s / photon_scale`, plus Gaussian read noise
`sigma_r`.  Writing `P = W^+`:

* **read noise**: `cov(x_hat) = sigma_r^2 / tau^2 * (W^T W)^-1`, so the mean
  per-view MSE is `sigma_r^2 / tau^2 * tr((W^T W)^-1) / N`.
* **photon noise**: `cov(x_hat) = 1/(tau * photon_scale) * P diag(W x_bar) P^T`
  (the variance of shot k is proportional to the signal it collected).  It is
  linear in the mean radiances `x_bar`, so averaging over pixels is exact.

For identity patterns `tr((W^T W)^-1) = N`.  For the Harwit-Sloane **S-matrix**
(`(N+1)/2` ones per row, `S^T S = (N+1)/4 (I + J)`) one finds
`tr((S^T S)^-1) = 4N^2/(N+1)^2`, i.e. an MSE gain of `(N+1)^2/(4N) ~ N/4` and
an **SNR gain `(N+1)/(2 sqrt N) ~ sqrt(N)/2` for read noise**.

For photon noise and a uniform scene, `W x_bar = x_bar (N+1)/2` for every
shot, so the per-view variance is `2N^2 x_bar/((N+1) T photon_scale)`,
which tends to **twice** the sequential variance `N x_bar/(T photon_scale)`: the S-matrix
*loses* a factor sqrt(2) in SNR (the extra light in each measurement carries
its own shot noise, which is spread over all views by the inversion).  This is
Schechner, Nayar and Belhumeur's finding that photon noise reduces or removes
the multiplexing advantage; Ratner and Schechner derive the optimal W within
fundamental limits and obtain sparser patterns (fewer open cells) in the
photon-limited regime.  Real cameras are in between, so the answer depends on
the ratio of read to photon noise (our "mixed" regime).

Multiplexing is independent of scene geometry: the model is linear in the
radiance of each aperture cell, so reflections, refractions and occlusions
(slides 31-37) pose no problem, unlike depth-based reconstructions.

### 2.3 What is the best W? (slide 30)

For read noise we minimise `tr((W^T W)^-1)` over 0/1 matrices; S-matrices are
optimal for n = 2^k-1 and (n = 3 mod 4 prime, Paley), and no construction is known for
arbitrary N such as 9 or 25.  For photon noise the objective is
`tr(P diag(W x_bar) P^T)` (Schechner-style).  `optimize_patterns` runs
simulated annealing over single bit flips with the chosen objective and returns
`(W, objective)`.  It is a heuristic: it reaches SNR gain 1.58 at N=9
(bound for an S-matrix of that size 1.67) and 2.38 at N=25 (2.60), but at
N=7/15 where S-matrices exist a single run stays 10-25 % short (use
`restarts` or just `s_matrix`).  For the photon objective it correctly
refuses to prefer dense patterns when started from the identity (gain 1.0).

### 2.4 Comparison with the plenoptic camera (slide 29)

The slide states, for N views in time T from an M-pixel sensor:
spatial resolution M/N (plenoptic) vs M (programmable aperture), and

| | plenoptic | PA no multiplexing | PA with multiplexing |
|---|---|---|---|
| photon-noise SNR (slide) | SNR_p | SNR_p / sqrt(N) | SNR_p / sqrt(2) |
| read-noise SNR (slide) | SNR_p | SNR_p / N | SNR_p / 2 |

Our derivation uses the plenoptic *sample* as reference: its pixel integrates
for the full T, so `var = x/(T ps)` (photon) and `sigma_r^2/T^2` (read).  Then:

| per output pixel, same pixel count as the sample | no mux | S-matrix |
|---|---|---|
| photon SNR / SNR_p | 1/sqrt(N) | about 1/sqrt(2N) |
| read SNR / SNR_p | 1/N | `(N+1)/(2 N sqrt N)` ~ 1/(2 sqrt N) |

The no-mux column **agrees** with the slide.  The mux column **differs by a
factor sqrt(N)**, exactly the gain from binning N full-resolution PA pixels
into one (the PA has N times more pixels, so at equal output resolution M/N
each output pixel averages N independent noisy pixels): multiplied by sqrt(N)
it gives photon `1/sqrt(2)` and read `~1/2`, the slide's numbers.  So the slide's mux
row is consistent only under the *binned* convention, while its no-mux row uses
the *per-pixel* convention.  Under one consistent convention:

| | per pixel (no binning) | binned to M/N |
|---|---|---|
| no mux | photon 1/sqrt(N), read 1/N | photon 1, read 1/sqrt(N) |
| mux | photon 1/sqrt(2N), read 1/(2 sqrt N) | photon 1/sqrt(2), read ~1/2 |

Consequences: (i) in the read-noise regime multiplexing gains about sqrt(N)/2
over sequential capture and, after binning, gets to within a factor 2 of the
plenoptic SNR while keeping an N times higher resolution *before* binning;
(ii) in the photon regime multiplexing does not help (it is sqrt(2) worse than
sequential for a uniform scene); the slide's "SNR_p/sqrt(2)" is better than
the no-mux row only because of the inconsistent convention.  The simulation
below (which uses the actual noise model and no binning) arbitrates in favour
of this derivation.  The plenoptic baseline also ignores that its pixels
integrate over T only if the lenslet is lossless and its resolution is M/N.

## 3. Algorithm

1. Choose W (`identity_patterns`, `s_matrix`/`hadamard_patterns`, `optimize_patterns`).
2. `capture(lf, W, ...)`: K shots of duration T/K with Poisson + Gaussian noise.
3. `demultiplex(Y, W, T/K)`: `x_hat = W^+ y / tau` for all pixels at once.
4. Optionally refocus, aperture-synthesise or estimate depth from `x_hat`.

## 4. Implementation notes

* `s_matrix(n)`: Sylvester Hadamard (n=2^k-1) or Paley circulant (n prime = 3 mod 4);
  `valid_s_sizes()` lists admissible n.  `hadamard_patterns(n)` falls back to the
  optimiser when no S-matrix exists.
* `expected_mse`, `snr_gain`, `theory_table` are closed forms; `simulate_snr`
  is Monte Carlo (`plenoptic=True` for the baseline via `capture_plenoptic`,
  which ignores the M/N resolution loss; see `resolution_penalty`).
* `demultiplex` supports a ridge regulariser (`ridge=`) and clipping; no
  clipping is used for the statistics, since clipping biases the noise.
* Cost: capture and inversion are one (K x N) matrix product over H*W*C columns.
* The sensor non-linearity, saturation, light leakage through an LCD, and
  per-pixel varying noise are not modelled.

## 5. Experiments

Settings: T = 1, `sigma_r` = 0.01, `photon_scale` = 1000 (mixed), synthetic LF
(`lf.synthetic`), 2 trials per point.  The Lego Knights LF (5x5, downsample 4) was available when the figures were made, so a real-data version exists too.

* `figures/aperture_patterns.png`: the 9 apertures of a 3x3 LF, one cell open
  (top) vs the optimised multiplexed patterns (bottom), as on slide 25.
* `figures/multiplex_pipeline.png`: a single-cell shot, its demultiplexed
  view, a multiplexed shot (blurred but clean) and its demultiplexed view,
  with the ground truth (slides 31-34).  The multiplexed result is cleaner
  (central view 20.3 dB vs 19.1 dB; all views 19.6 vs 19.2 dB in the mixed regime,
  where the gain is small, matching the table below).
* `figures/multiplex_pipeline_lego.png`: same pipeline on the real Stanford Lego Knights 5x5 LF (N=25, same noise model): central view 12.6 dB sequential vs 14.6 dB multiplexed; all views 11.1 vs 13.6 dB.
* `figures/multiplex_snr_vs_N.png`: SNR gain over sequential capture vs N in
  the three regimes, theory (lines) and simulation (markers).  Theory and
  simulation agree to better than 1 %.
* `figures/multiplex_refocus.png`: refocusing (alpha = 1.5) the demultiplexed
  5x5 LF: refocusing averages views and removes much noise (27.2 dB sequential,
  30.5 dB multiplexed vs. 11.3 / 14.6 dB per view).  For the S-matrix the
  noise of the *sum* of all views is reduced even more strongly, because
  `1^T W^-1 = (2/(N+1)) 1^T`.

SNR gain over sequential capture (amplitude; theory / simulation):

| regime | N | S-matrix | optimised | plenoptic |
|---|---|---|---|---|
| read | 9 | n/a | 1.585 / 1.583 | 9.00 / 9.00 |
| read | 25 | n/a | 2.377 / 2.378 | 25.0 / 25.0 |
| read | 7 / 15 | 1.512 / 1.510, 2.066 / 2.066 | 1.32, 2.02 | 7, 15 |
| photon | 9 | n/a | 0.727 / 0.726 | 3.00 / 2.99 |
| photon | 25 | n/a | 0.668 / 0.666 | 5.00 / 4.99 |
| mixed | 9 | n/a | 1.064 / 1.061 | 4.74 / 4.74 |
| mixed | 25 | n/a | 1.437 / 1.440 | 11.70 / 11.71 |

The plenoptic column is the same-pixel-count baseline from 2.4 (no binning,
no resolution penalty).  The S-matrix read gain matches `(N+1)/(2 sqrt N)`
(N=7: 1.512, N=15: 2.066).

## 6. Try it

```python
import lf, numpy as np
from lf import multiplex as mx
L = lf.synthetic.synthetic_light_field(n_views=3, size=96)
W, f = mx.optimize_patterns(9, rng=0)
Y = mx.capture(L, W, photon_scale=None, read_noise=0.01, rng=0)
est = mx.demultiplex(Y, W, 1/9)
print(mx.snr_gain(W), mx.theory_table(9)['read'])
```
`python scripts/make_figures_multiplex.py` regenerates all figures and prints the table.

## 7. Limitations and extensions

* Photon noise defeats multiplexing; with sparse-signal scenes the picture
  changes (the objective depends on `x_bar`), so adapt W to the signal.
* Sequential capture: the scene must be static during the N (or K) shots; moving
  subjects break the linear model (slide 42).  Light leakage through an LCD
  (imperfect on/off contrast) makes W non-binary and must be calibrated.
* Prototypes (slides 38-41): a paper scroll with printed patterns moved by a
  motor, and an LCD array; the LCD loses light to polarisers.
* Future work (slide 42): compressive sensing, using fewer than N shots
  (K < N) with sparse priors; `capture`/`demultiplex` already accept K != N.
* Connection to HDR: Nayar & Branzoi's adaptive dynamic-range camera also
  modulates light per pixel in time (slide 18); the same linear-inversion view applies.
* The annealer is heuristic; try restarts, tabu search or Ratner-Schechner's
  photon-limited constraint set.

## 8. References

* Liang, Lin, Wong, Liu, Chen. Programmable Aperture Photography: Multiplexed Light Field Acquisition. ACM SIGGRAPH 2008.
* Harwit, Sloane. Hadamard Transform Optics. Academic Press, 1979.
* Schechner, Nayar, Belhumeur. Multiplexing for Optimal Lighting. IEEE TPAMI 29(8), 2007.
* Ratner, Schechner. Illumination Multiplexing within Fundamental Limits. CVPR 2007.
* Wetzstein, Ihrke, Heidrich. On Plenoptic Multiplexing and Reconstruction. IJCV 101(2), 2013.
* Veeraraghavan, Raskar, Agrawal, Mohan, Tumblin. Dappled Photography. ACM SIGGRAPH 2007.
* Nayar, Branzoi. Adaptive Dynamic Range Imaging: Optical Control of Pixel Exposures over Space and Time. ICCV 2003.
