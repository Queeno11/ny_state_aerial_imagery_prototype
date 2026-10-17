# Rehearsal plan

30 main slides plus title; 19 backup slides. Target: 35.00 minutes before discussion.

| Slide | Topic | Minutes | Cumulative |
|---|---|---:|---:|
| 1 | Can imagery fill gaps between surveys? | 1.00 | 1.25 |
| 2 | Good rankings; economic change remains difficult | 1.00 | 2.25 |
| 3 | Four questions in one research agenda | 0.50 | 2.75 |
| 4 | What existing imagery models already deliver | 0.75 | 3.50 |
| 5 | The US measurement setting | 1.00 | 4.50 |
| 6 | The label is a constructed wealth measure | 1.25 | 5.75 |
| 7 | Similar structures can carry different dollar labels | 1.00 | 6.75 |
| 8 | Image changes need not be economic changes | 0.75 | 7.50 |
| 9 | A common index with a local measurement scale | 1.75 | 9.25 |
| 10 | The central economic restriction is invariance | 1.00 | 10.25 |
| 11 | Stable pairs add a second source of information | 1.25 | 11.50 |
| 12 | Within-cell comparisons eliminate the local scale | 1.25 | 12.75 |
| 13 | The criterion combines ranking and stability | 1.50 | 14.25 |
| 14 | Population recovery has a price for imperfect pairs | 2.00 | 16.25 |
| 15 | More cities remove sampling error, not every error | 1.50 | 17.75 |
| 16 | Cross-city alignment is still an open problem | 0.75 | 18.50 |
| 17 | Monte Carlo: robustness to non-comparable labels | 1.50 | 20.00 |
| 18 | Monte Carlo: what do temporal pairs add? | 1.50 | 21.50 |
| 19 | Monte Carlo: does the gain survive flexibility? | 1.50 | 23.00 |
| 20 | The current US run approximates the framework | 1.25 | 24.25 |
| 21 | Generalization tests separate cities, dates and sensors | 1.00 | 25.25 |
| 22 | Tract rankings transfer to held-out cities | 1.00 | 26.25 |
| 23 | The ordinal run improves rankings in 35 of 36 cities | 0.75 | 27.00 |
| 24 | Persistence is high; detecting change is harder | 1.00 | 28.00 |
| 25 | Construction tests responsiveness of the fitted index | 1.00 | 29.00 |
| 26 | NYC scores rise around construction | 1.25 | 30.25 |
| 27 | Fine spatial detail, with tract-level validation | 0.75 | 31.00 |
| 28 | Held-out tract changes remain poorly predicted | 1.75 | 32.75 |
| 29 | Temporal measurement requires more than stable images | 1.00 | 33.75 |
| 30 | What is established, and what comes next | 1.25 | 35.00 |

If discussion is heavy: skip the literature slide (4), shorten the image comparison (8), and move convergence (15) to backup. Keep the findings preview, the estimand, the current implementation gap, and the final change result.

Main-slide speaker notes are embedded in main.tex. Compile a presenter version by defining \shownotes before inputting main.tex.

## Tomorrow: insert the pending figures

Place the five MC PDFs in figures/mc using the exact filenames below. The deck includes them automatically on recompilation:

- mc1_correct.pdf
- mc1_misspecified.pdf
- mc3_linear_links.pdf
- mc3_nn_linear.pdf
- mc3_nn_quadratic.pdf

Place six puzzle PNGs in figures/puzzle: nyc_rich.png, nyc_mid.png, nyc_poor.png, chi_rich.png, chi_mid.png, chi_poor.png. No invented wealth labels are shown. Add actual tract/year labels if they support the comparison.

Before presenting the new MC results, update notes with actual DGP parameters, repetitions, uncertainty, and observed takeaways. The current MC slides pose questions and make no unsupported numerical claims.