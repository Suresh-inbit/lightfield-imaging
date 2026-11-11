# 00 - Light fields: background and the map of this project

## 1. In one paragraph
A conventional camera integrates all the rays arriving at a pixel over the lens
aperture, so the *direction* of light is lost at capture time (lecture 15,
slides 5-6: a "dead picture" with no post-capture control).  A light-field
camera records radiance as a function of position *and* direction.  With that
4-D data in hand, perspective change, refocusing, depth-of-field control, depth
estimation and extended depth of field become *computations* instead of
decisions made before pressing the shutter (slides 2-4, 14-15).  This project
implements each of those computations from first principles, documents them,
and adds an exploration of high-dynamic-range light fields.

## 2. Theory

### 2.1 From the plenoptic function to the 4-D light field
The plenoptic function (Adelson & Bergen 1991) gives the radiance along every
ray: position (3), direction (2), wavelength and time.  In free space radiance
is constant along a ray, so for a fixed wavelength and instant the function is
4-D (slide 19).  Levoy & Hanrahan (1996) and Gortler et al. (1996) parameterise
a ray by its intersections with two parallel planes: `L(u, v, s, t)`.

Inside a camera (slide 24) the natural planes are the **lens/aperture plane**
`(u, v)` and the **sensor plane** `(s, t)`.  Fixing `(u, v)` and varying
`(s, t)` gives a *sub-aperture image*: a pinhole-like photograph taken through
one small part of the lens (slides 25-27: each sub-square of the 4-D picture is
the set of rays leaving one point on the lens).  Fixing `(s, t)` and varying
`(u, v)` gives the *image of the aperture* seen from one sensor point.

In this code base the array index order is `L[u, v, s, t]` and the sub-aperture
image for aperture position `(u, v)` is `lf.views[u, v]` (see
`docs/conventions.md`).

### 2.2 A conventional photograph is an angular integral
The irradiance at a sensor point is `E(s, t) = ∬ L(u, v, s, t) du dv` (lecture
16, slide 11).  The integral is not invertible: angular information is gone,
so neither the viewpoint nor the focal plane can be changed afterwards.  Every
light-field camera is a trick to *avoid* performing that integral in the optics.

### 2.3 Parallax, disparity and epipolar-plane images
Move the viewpoint by `Δu` along the aperture plane and a scene point moves in
the image by `d · Δu`, where the **disparity** `d` depends only on the point's
depth (zero for points on the plane the light field is rectified to).  Slicing
the light field along `(v, t)` for fixed `(u, s)` produces an *epipolar-plane
image* (EPI) in which each scene point traces a straight line of slope `d`
(Bolles, Baker & Marimont 1987).  Every algorithm here can be read as an
operation on EPI slopes:

| Operation | Slide | What it does to the EPI |
|---|---|---|
| View change | 61-64 | pick one row |
| Refocus on disparity α | 45-56 | shear by α, then sum rows |
| Stop down | 43 | sum fewer rows (central ones) |
| Depth estimation | 15 | measure the slope |
| All-in-focus | 58 | per pixel, pick the shear that aligns |

### 2.4 How light fields are captured (slide 16)
* **Camera arrays / gantries** (Stanford camera array; the Stanford gantry data
  used here): many conventional cameras, large baseline, full spatial
  resolution per view, bulky.
* **Plenoptic / micro-lens cameras** (Adelson & Wang 1992; Ng et al. 2005;
  Lytro): a micro-lens array at the sensor plane spreads the rays from each
  image point over the pixels beneath the micro-lens.  Single shot; spatial
  resolution is traded for angular resolution.  `docs/02_plenoptic_camera.md`.
* **Programmable aperture** (Liang et al. 2008): open a different part of the
  aperture in each of N exposures; full sensor resolution; sequential; made
  practical by multiplexing.  `docs/08_programmable_aperture.md`.
* **Mask-based / dappled photography** (Veeraraghavan et al. 2007): a cosine
  mask near the sensor heterodynes angular information into the spatial
  spectrum.  Not implemented here, but the multiplexing analysis in doc 08
  applies to it as well (Wetzstein, Ihrke & Heidrich 2013).

## 3. Map of this project

| Doc | Topic | Module | Lecture |
|---|---|---|---|
| 01 | Datasets and loaders | `lf/io.py`, `scripts/download_data.py` | - |
| 02 | Plenoptic camera: optics, raw sensor image, sub-aperture extraction | `lf/plenoptic.py` | 15 |
| 03 | Digital refocusing (shift-and-add, Ng's α') | `lf/refocus.py` | 15 |
| 04 | Digital stopping down / depth of field | `lf/aperture.py` | 15 |
| 05 | Perspective shift, view synthesis | `lf/aperture.py` | 15 |
| 06 | Extended depth of field (all-in-focus) | `lf/edof.py` | 15 |
| 07 | Depth from defocus, correspondence and EPI slopes | `lf/depth.py` | 15 |
| 08 | Programmable aperture and multiplexed acquisition | `lf/multiplex.py` | 16 |
| 09 | HDR light fields | `lf/hdr.py` | 16 (background slides) |

The interactive application (`app/streamlit_app.py`) exposes every method with
sliders; the notebooks in `notebooks/` walk through them in order.

## 4. Notation used throughout

| Symbol | Meaning |
|---|---|
| `U × V` | number of angular samples (views) |
| `H × W` | spatial resolution of one view |
| `(u0, v0)` | central view |
| `d` | disparity in pixels per angular step; `d > 0` nearer than the rectification plane |
| `α` | the disparity a refocused image is focused on |
| `N = U·V` | number of views (also the number of exposures of a programmable aperture) |
| `T` | total exposure budget |

## 5. References
* E. H. Adelson, J. R. Bergen, "The plenoptic function and the elements of early vision", in *Computational Models of Visual Processing*, MIT Press, 1991.
* E. H. Adelson, J. Y. A. Wang, "Single lens stereo with a plenoptic camera", IEEE TPAMI 14(2), 1992.
* M. Levoy, P. Hanrahan, "Light field rendering", SIGGRAPH 1996.
* S. J. Gortler, R. Grzeszczuk, R. Szeliski, M. F. Cohen, "The Lumigraph", SIGGRAPH 1996.
* R. C. Bolles, H. H. Baker, D. H. Marimont, "Epipolar-plane image analysis: an approach to determining structure from motion", IJCV 1(1), 1987.
* R. Ng, M. Levoy, M. Bredif, G. Duval, M. Horowitz, P. Hanrahan, "Light field photography with a hand-held plenoptic camera", Stanford Tech Report CTSR 2005-02.
* C.-K. Liang, T.-H. Lin, B.-Y. Wong, C. Liu, H. H. Chen, "Programmable aperture photography: multiplexed light field acquisition", ACM TOG 27(3) (SIGGRAPH), 2008.
* A. Veeraraghavan, R. Raskar, A. Agrawal, A. Mohan, J. Tumblin, "Dappled photography: mask enhanced cameras for heterodyned light fields and coded aperture refocusing", SIGGRAPH 2007.
* B. Wilburn et al., "High performance imaging using large camera arrays", SIGGRAPH 2005 (the Stanford camera array).
* G. Wetzstein, I. Ihrke, W. Heidrich, "On plenoptic multiplexing and reconstruction", IJCV 101, 2013.
