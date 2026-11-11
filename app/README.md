# Light-field explorer (Streamlit)

```bash
cd lightfield
streamlit run app/streamlit_app.py
```

Sidebar: dataset (only those found by `lf.io.available()`, plus `synthetic`), spatial downsample
(Stanford data is never loaded below 4), angular crop, synthetic size/views, and a summary of the
loaded light field (angular, spatial, memory).  Loaded light fields are cached
(`st.cache_resource`); focal stacks, depth maps, captures and GIFs are cached (`st.cache_data`).
Heavy computations (depth, all-in-focus, HDR) run when you press their button.

## Tour

1. **Explore** - view grid, (u, v) picker, horizontal/vertical EPIs with slope lines for chosen disparities.
2. **Plenoptic camera** - raw lenslet mosaic with zoom, sub-aperture extraction picture,
   conventional vs stopped-down photo, optical design calculator and the slide-14 resolution table.
3. **Refocus** - alpha slider, method, aperture shape/radius, ROI sharpness, focal-stack strip, sweep GIF.
4. **Perspective** - fractional (u, v) view, corner views, perspective-path GIF.
5. **Depth and EDoF** - four disparity estimators, confidence, ground-truth error (synthetic),
   all-in-focus and the slide-58 triptych.
6. **Programmable aperture** - patterns, simulated shots, demultiplexing PSNR, theory table, refocus.
7. **HDR** - synthetic HDR scene or pseudo-HDR from the dataset, exposure patterns, strategies A/C/D,
   tonemapping, log-RMSE, LDR vs HDR refocus.

Each tab has a "How this works" box and a "Full note" box that renders the matching `docs/NN_*.md`.

Tests: `python -m pytest tests/test_app.py -q` (uses `streamlit.testing.v1.AppTest`).
