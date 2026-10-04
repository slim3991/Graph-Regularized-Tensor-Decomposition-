# Graph-Regularized Tensor Decomposition

Graph-regularized canonical polyadic (CP) decomposition for real NumPy tensors.
This project is turning code from a master's thesis into a small Python library.
It provides an alternating least squares (ALS) solver, a conjugate gradient
(CG) solver for Sylvester equations, and a scikit-learn-compatible estimator.

The current implementation supports tensors with two or more dimensions and
precomputed graph Laplacians, with a separate regularization strength per mode.

## Setup

Requires Python 3.13 or newer. From the repository root, install dependencies
with `uv`:

```sh
uv sync
```

Alternatively, create a virtual environment and install dependencies with pip:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install 'numpy>=2.5.3' 'tensorly>=0.9.0' 'scikit-learn>=1.6'
```

Run examples from the repository root. The package layout is still under
development; the current import path is `src.GRCP`.

## Quick start

This example connects adjacent entries along each tensor mode with a path graph.
For real data, supply graphs that describe relationships between entries in
each dimension.

```python
import numpy as np

from src.GRCP import MyGRTenDecomp


def path_laplacian(size):
    adjacency = np.zeros((size, size))
    indices = np.arange(size - 1)
    adjacency[indices, indices + 1] = 1
    adjacency[indices + 1, indices] = 1
    return np.diag(adjacency.sum(axis=1)) - adjacency


X = np.ones((4, 3, 2))
laps = [path_laplacian(size) for size in X.shape]

model = MyGRTenDecomp(
    rank=1,
    laps=laps,
    regularization_strengths=[0.01, 0.02, 0.03],
    n_iter=500,
    tol=1e-6,
    random_state=42,
)
reconstruction = model.fit_transform(X)
residual = model.residuals(X)

print("Converged:", model.converged_)
print("Iterations:", model.n_iter_)
print("Residual norm:", np.linalg.norm(residual))
```

`transform(X)` returns a copy of the fitted training reconstruction and requires
the training tensor's shape. It does not decompose new observations.
`residuals(X)` subtracts that reconstruction from the supplied tensor.

## Objective and normalization

For a nonzero tensor, ALS normalizes the input to unit Frobenius norm and
minimizes

```text
0.5 * ||X / ||X||_F - CP(U_0, ..., U_{d-1})||_F^2
+ 0.5 * sum_m lambda_m * trace(U_m.T @ L_m @ U_m)
```

Here, `U_m` is the factor matrix for mode `m`, `L_m` is its graph Laplacian,
and `lambda_m` is its regularization strength. The graph term penalizes variation
between connected entries in a factor matrix.

Factor columns are not rescaled during optimization, because rescaling changes
the graph penalty. Returned CP weights restore the input tensor's scale:
reconstructions and residuals are in the original units, while objective
diagnostics refer to the normalized tensor. Zero tensors return an exact zero
solution.

## Estimator parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rank` | `5` | Positive integer number of CP components. |
| `laps` | `None` | One finite, symmetric positive-semidefinite matrix per mode, matching that mode's size. `None` entries disable individual graph penalties. |
| `regularization_strengths` | `None` | One finite nonnegative strength per mode. `None` uses ones; zero disables a mode's penalty. |
| `n_iter` | `100` | Maximum number of ALS sweeps. |
| `tol` | `1e-6` | Relative tolerance for change in the full objective. |
| `random_state` | `None` | Seed for reproducible initialization. |

Set `laps=None` to run without graph penalties. Input tensors must contain finite
real numbers and have at least two nonempty dimensions. Integer inputs are
converted to floating point. Neither input arrays nor Laplacians are modified.

After fitting, inspect `cp_tensor_`, `X_hat_`, `E_`, `laps_`, `tensor_shape_`,
`tensor_norm_`, `converged_`, `n_iter_`, and `objective_history_`.
`diagnostics_` also contains reconstruction-error and graph-penalty histories,
the solver status, and CG diagnostics for regularized updates.

## Direct solver use

`graph_regularized_als` returns `(cp_tensor, residual)` by default. Request
diagnostics with `return_info=True`:

```python
import tensorly as tl

from src.GRCP import graph_regularized_als

cp_tensor, residual, info = graph_regularized_als(
    X,
    rank=1,
    laps=laps,
    regularization_strengths=[0.01, 0.02, 0.03],
    n_iter=500,
    random_state=42,
    return_info=True,
)
reconstruction = tl.cp_to_tensor(cp_tensor)
```

`global_cg_sylvester` solves `A @ solution + solution @ B = C`:

```python
from src.GRCP import global_cg_sylvester

A = np.diag([1.0, 2.0])
B = np.diag([3.0, 4.0])
C = np.ones((2, 2))

solution, info = global_cg_sylvester(
    A, B, C, tol=1e-8, atol=1e-12, return_info=True
)
```

A and B must be finite real symmetric matrices whose summed Sylvester operator
is positive semidefinite. Positive definiteness guarantees a unique solution;
consistent singular systems may also converge. CG stops when the actual
residual norm is at most `max(atol, tol * ||C||_F)`. It preserves the supplied
initial guess `x0`. Without `return_info=True`, it returns only the solution.

Both solvers emit a `RuntimeWarning` on nonconvergence and provide status
information when diagnostics are requested. Check convergence before relying
on a returned result.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Or, using an existing virtual environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover reconstruction, residuals, per-mode regularization, graph
smoothing, objective decrease, normalization, reproducibility, zero tensors,
higher tensor orders, estimator validation, and CG agreement with SciPy,
including singular systems and iteration exhaustion.

## Current limitations

- Graph construction from `laplacian_parameters` and thresholding through
  `local_threshold` are not implemented. Requesting either raises
  `NotImplementedError`.
- ALS solves a nonconvex problem and does not guarantee a global optimum.
  Laplacian nullspaces and unpenalized modes can permit factor-scale
  degeneracies; normalizing the input does not remove them.
- Matrices are dense, and validation uses eigenvalue calculations. Large sparse
  graphs need a separate implementation.
- An installable package and a stable public API are still being developed.

See [the implementation notes](docs/grcp-cleanup.md) for more details.

## License

See [LICENSE](LICENSE).
