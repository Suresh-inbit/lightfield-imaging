# 05 - Moving the observer (perspective shift)

## 1. In one paragraph
Each sub-aperture image is a pinhole view from a slightly different point on the lens. Choosing which window of pixels under every microlens to extract moves the virtual camera within the aperture (slides 41, 61-64). Between grid positions we can interpolate; the result is plausible for small baselines but ghosts when disparities are large.

## 2. Theory
View `(u,v)` is the 2-D slice `L(u,v,.,.)`. A scene point of disparity `d` moves `d` pixels per angular step, so nearer points move more than farther ones: parallax. At fractional `(u,v)` the ideal image is `L(u,v,.,.)`, which we approximate by bilinear blending of the 4 neighbours:
`I = (1-a)(1-b) L_00 + (1-a) b L_01 + a(1-b) L_10 + a b L_11`.
A point of disparity `d` then shows up in two places `d` pixels apart, weighted `(1-a)` and `a`. This is "angular aliasing"; it is invisible when `d <= 1` px and becomes visible double edges beyond that. Correct synthesis needs per-pixel depth (depth-image-based rendering): warp each neighbour by `d(s,t) * (du, dv)` before blending.

**Forward/backward (slide 64).** Moving the camera along the optical axis requires a change in parallax per depth; `dolly` is only the crude approximation "refocus then zoom about the centre", exact for a flat scene at the focal plane.

## 3. Algorithm
1. Choose a path of `(u,v)` positions (circle / row / column / diagonal) of radius up to the grid half-size.
2. For each position blend the 4 nearest views.

## 4. Implementation notes
* `synthesize_view(lf, u, v)` clips to the grid; integer `(u,v)` returns exactly `lf.view(u,v)`.
* `perspective_sequence(lf, path, n, radius)` returns a list of images; paths are closed so GIFs loop.
* `dolly(lf, alpha, scale)` uses cv2 `getRotationMatrix2D` zoom with replicated borders.

## 5. Experiments
* `docs/figures/perspective_shift_strip.png` - four corner views and two synthesized intermediates. Track the gravel disc (near) against the grass (far): they move in opposite directions across the strip; the intermediates look like in-betweens with slight doubling of the gravel edge.
* `docs/figures/perspective_shift.gif` - a circular camera path; the gravel orbits more than the text, which orbits more than the grass.
* Both figures use the synthetic scene; no real-data perspective figure is generated.
* Test: synthesize at integer `(u,v)` equals the stored view; at `(2.5, 3)` equals the mean of the two neighbours.

## 6. Try it
```python
from lf import aperture as A, synthetic
L = synthetic.synthetic_light_field(9, 160)
frames = A.perspective_sequence(L, 'circle', n=24)
```

## 7. Limitations and extensions
* Ghosting for large disparity (above); the baseline is limited to the aperture (a few mm for a Lytro), so only small parallax is possible.
* Extension: estimate disparity (`lf.depth`) and render with warping + inpainting; compare with the blended result.

## 8. References
* Ng et al., Stanford TR 2005-02 / SIGGRAPH 2005.
* Levoy, Hanrahan, "Light field rendering", SIGGRAPH 1996 (view interpolation by bilinear blending in (u,v)).
* Isaksen, McMillan, Gortler, "Dynamically reparameterized light fields", SIGGRAPH 2000.
