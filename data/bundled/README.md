# Bundled sample light field

`lego_knights_17x17_ds4.png` holds all 17 x 17 views of the **Lego Knights** light field from the Stanford (New)
Light Field Archive (V. Vaish, A. Adams, Stanford Computer Graphics Laboratory, <https://lightfield.stanford.edu/lfs.html>;
source <https://lightfield.stanford.edu/data/lego_lf/rectified.zip>), box-downsampled 4x to 256 x 256 px per view and
tiled row-major. The views are stored in this project's orientation (nearer objects have positive disparity).

It is included so the app has a real light field at start-up without a 373 MB download.
Load it with `lf.io.load('lego', downsample=4, angular_crop=9)`. If the full-resolution set has been downloaded
(`python scripts/download_data.py lego`), the loader uses that instead.
