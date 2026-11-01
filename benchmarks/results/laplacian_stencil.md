# Laplacian stencil: NumPy versus C++

Time per call of the stencil alone, in microseconds. Ghost filling and conjugate gradient are not included, so these are not whole-solver speedups.

| Grid | NumPy | C++, no threads | C++, final | Speedup vs NumPy |
|---|---|---|---|---|
| 16³ | 24.9 | 1.78 | 1.87 | 13.3× |
| 24³ | 75.1 | 5.35 | 5.52 | 13.6× |
| 32³ | 163 | 12.8 | 13.1 | 12.4× |
| 48³ | 507 | 38.3 | 39.1 | 13.0× |
| 64³ | 1,230 | 97.6 | 72.6 | 17.0× |
| 100³ | 4,350 | 393 | 167 | 26.1× |
| 128³ | 9,420 | not run | 286 | 32.9× |
| 200³ | 37,000 | 2,940 | 984 | 37.7× |
| 256³ | 77,500 | not run | 2,340 | 33.1× |
| 400³ | 291,000 | not run | 8,550 | 34.0× |
| 512³ | 630,000 | not run | 20,300 | 31.0× |

"Final" is commit `a022ec9`: one thread below 200,000 points, and 4 threads from 64³ up. Up to 48³ it runs the same code as "no threads", so the differences there are noise.

The speedup uses the "final" column. The 128³, 256³, 400³ and 512³ rows come from a separate run from the others.

## Method

- Machine: Apple M4 Pro, one process.
- Each value is the median of 7 samples, and each sample averages many calls.
- Random 3D array of float64, with one ghost layer and `dx = dy = dz = 1/n`.
- NumPy runs the same operations as the solver's old `_apply_laplacian`, without ghost filling.

## Whole projection

From `benchmarks/pressure_poisson.py`, with `_apply_laplacian` calling the C++ kernel. Median of 3 runs on one process.

| Grid | NumPy | C++ | Speedup |
|---|---|---|---|
| 50³ | 0.13 s | 0.04 s | about 3× |
| 100³ | 1.89 s | 0.55 s | about 3.4× |
