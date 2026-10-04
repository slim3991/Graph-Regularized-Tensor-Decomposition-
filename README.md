# Graph-Regularized Tensor Decomposition

A small Python library for graph-regularized canonical polyadic (CP)
decomposition of real, three-mode NumPy tensors. It includes an alternating
least squares (ALS) solver, a conjugate gradient (CG) Sylvester solver, and a
wrapper following sklearn estimator conventions.

This is a small library adapted from my masters thesis

## Installation

Requires Python 3.13 or newer. From the repository root:

```sh
uv sync
```

Alternatively, install into your Python environment:

```sh
python -m pip install .
```

For an editable development installation, use `python -m pip install -e .`.
The import name is `grcp`.

## Quick start

Supply one Laplacian per tensor mode. Dense NumPy matrices and SciPy sparse
matrices are supported. Use `None` to disable a mode's graph penalty and scale
Laplacians to set regularization strength. Graph construction is the caller's
responsibility.

```python
import numpy as np
from grcp import MyGRTenDecomp


def path_laplacian(size):
    adjacency = np.diag(np.ones(size - 1), 1)
    adjacency += adjacency.T
    return np.diag(adjacency.sum(axis=1)) - adjacency


X = np.einsum("i,j,k->ijk", [1., 3., 1., 3.], [1., 2., 3.], [1., 2.])
model = MyGRTenDecomp(
    laplacians=[0.1 * path_laplacian(4), None, None],
    rank=1,
    n_iter=20,
    tol=1e-6,
)
reconstruction = model.fit_transform(X)
residual = model.residuals(X)
print("Residual norm:", np.linalg.norm(residual))
```

`fit(X, y=None)` ignores `y` and returns the estimator. The wrapper checks for
finite real input and converts it to floating point. `transform(X)` requires
the fitted tensor's shape and returns a copy of the fitted reconstruction.
It does not decompose new observations. `residuals(X)` subtracts that
reconstruction from `X`. Ordinary sample-based cross-validation is not an
established workflow for this whole-tensor estimator.

## Objective and scaling

ALS uses the original input scale and minimizes

```text
0.5 * ||X - CP(U_0, U_1, U_2)||_F^2
+ 0.5 * sum_m trace(U_m.T @ L_m @ U_m)
```

Here, each supplied `L_m` includes its regularization strength; `None` means
no penalty. Initial CP weights are absorbed into the first factor, then weights
stay at one. Factors are not normalized during optimization because rescaling
changes the graph penalty. No ridge penalty is added. Unregularized mode
updates use least squares, and regularized updates use CG.

After every ALS sweep, the solver checks relative change in the full objective
against `tol`, with a machine-epsilon floor for the comparison scale.

## Estimator parameters and outputs

| Parameter | Default | Meaning |
| --- | --- | --- |
| `laplacians` | Required | Three matrices matching mode sizes, or `None` entries. |
| `rank` | `5` | Number of CP components. |
| `n_iter` | `20` | Maximum number of ALS sweeps. |
| `tol` | `1e-6` | Relative objective-change tolerance; also passed to initialization and CG. |

Parameters after `laplacians` are keyword-only. Fitted attributes are
`cp_tensor_`, `X_hat_`, `E_`, `laps_`, and `tensor_shape_`. There are currently
no convergence-status or objective-history attributes.

## Direct solver use

```python
import numpy as np
import tensorly as tl
from grcp import graph_regularized_als, global_cg_sylvester

X = np.arange(1., 25.).reshape(3, 4, 2)
cp_tensor, residual = graph_regularized_als(
    X, rank=2, laplacians=[None, None, None], n_iter=20,
)
reconstruction = tl.cp_to_tensor(cp_tensor)

A = np.diag([1., 2.])
B = np.diag([3., 4.])
C = np.ones((2, 2))
solution = global_cg_sylvester(A, B, C, tol=1e-8)
np.testing.assert_allclose(A @ solution + solution @ B, C)
```

CG solves `A @ solution + solution @ B = C`. Use compatible floating-point
arrays with symmetric matrices defining a positive-definite Sylvester operator;
consistent singular positive-semidefinite systems may also converge. Sparse
`A` is supported. These mathematical conditions are not validated automatically.
CG preserves `x0`, tests relative residual against `tol * ||C||_F`, and warns
if it breaks down or exhausts `max_iter`. ALS does not warn on sweep exhaustion.

## Development and tests

```sh
uv run python -m unittest discover -s tests -v
uv build
```

Tests cover reconstruction, residuals, graph smoothing, dense/sparse Laplacians,
objective decrease, fixed weights, singular unregularized updates, wrapper
validation and cloning, and CG agreement with SciPy and convergence edge cases.

## Current limitations

- Only three-mode tensors are supported, with dense tensors and factor matrices.
- Laplacian validation checks count and shape, not finiteness, symmetry, or
  positive semidefiniteness. Parameter validation is incomplete.
- ALS is nonconvex. Laplacian nullspaces and unregularized modes can permit
  factor-scale degeneracies; fixed weights do not eliminate them.
- Initialization is random; there is no `random_state` parameter yet.
- CG integer inputs, zero iteration limits, and its fixed breakdown threshold
  for small-scale systems still need attention.
- The public API is under development.

## License

Apache 2.0; see [LICENSE](LICENSE).
