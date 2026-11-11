| method | shots | LF log-RMSE | refocus a=0 | refocus a=1.5 |
|---|---|---|---|---|
| A checker, gt depth | 1 | 0.487 | 0.193 | 0.696 |
| A rows, gt depth | 1 | 0.613 | 0.226 | 0.880 |
| A radial, gt depth | 1 | 0.816 | 0.364 | 1.058 |
| A random, gt depth | 1 | 0.489 | 0.205 | 0.701 |
| A checker, no depth | 1 | 1.041 | 1.376 | 0.580 |
| A checker, refocused-domain merge | 1 | - | 0.786 | 0.367 |
| A checker, estimated depth (MAE 1.01px) | 1 | 0.428 | 0.469 | 0.249 |
| C temporal bracket (3 shots) | 3 | 0.248 | 0.155 | 0.151 |
| single exposure x0.03125 | 1 | 2.985 | 1.348 | 1.330 |
| single exposure x1 | 1 | 1.304 | 1.274 | 1.101 |
| single exposure x32 | 1 | 2.687 | 2.954 | 2.763 |
| D aperture area 1/3/9 cells (3 shots) | 3 | - | 1.408 | 0.914 |
| B refocus LDR (exp 1, display avg) | 1 | - | 1.434 | 1.148 |
| B refocus LDR linearised avg | 1 | - | 1.274 | 1.101 |
| B refocus HDR (from C) in radiance | 3 | - | 0.155 | 0.151 |
