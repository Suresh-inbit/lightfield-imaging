# Bundled sample light field

`lytro_people_9x9.png` holds the central 9 x 9 sub-aperture views (375 x 540 px each, tiled row-major, lossless,
no gain applied) of `people_1_eslf.png` from the Stanford Lytro Light Field Archive
(A. S. Raj, M. Lowney, R. Shah, G. Wetzstein, 2016, <https://lightfields.stanford.edu/LF2016.html>;
source file <https://lightfields.stanford.edu/images/people/raw/people_1_eslf.png>).

It is included so the app has a real plenoptic-camera light field at start-up without a 180 MB download.
Load it with `lf.io.load('lytro_people')`. If the full 14 x 14 ESLF has been downloaded
(`python scripts/download_data.py lytro_people`), the loader uses that instead.
