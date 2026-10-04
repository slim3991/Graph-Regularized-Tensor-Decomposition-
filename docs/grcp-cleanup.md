# Graph-regularized CP

`src/GRCP.py` provides `MyGRTenDecomp`, `graph_regularized_als`, and
`global_cg_sylvester`. Tensors may have two or more nonempty dimensions.
Inputs must be finite real arrays. Integer arrays are converted to floating point.

## Objective and normalization

ALS divides the input tensor by its Frobenius norm and minimizes

```text
0.5 * ||X / ||X||_F - CP(U_0, ..., U_d)||_F^2
+ 0.5 * sum_m lambda_m * trace(U_m.T @ L_m @ U_m)
```

`regularization_strengths` supplies one finite nonnegative strength per mode.
It defaults to ones. `laps` supplies one symmetric positive-semidefinite matrix
per mode, or None to disable that mode's penalty. Strength zero also disables
that penalty. Graph construction is not implemented: provide matrices directly.
Neither Laplacians nor factor columns are normalized automatically.

CP weights remain one during optimization. Factor rescaling was removed because
it changes the graph penalty. Returned CP weights carry the input norm, so
reconstructions and residuals are in the original input units. Diagnostics
refer to the normalized objective. A zero tensor returns an exact zero solution.

```python
from src.GRCP import MyGRTenDecomp

model = MyGRTenDecomp(
    rank=3,
    laps=[L0, L1, L2],
    regularization_strengths=[0.1, 0.5, 0.0],
    n_iter=500,
    random_state=42,
).fit(X)
reconstruction = model.transform(X)
residual = model.residuals(X)
print(model.converged_, model.n_iter_, model.objective_history_)
```

`transform` returns a copy of the training reconstruction and requires the
training shape. It does not fit new observations. Graph-construction parameters
and local thresholding raise NotImplementedError when requested.

## Solver behavior

ALS supports arbitrary tensor order, uses least squares for unregularized mode
updates, and CG for regularized Sylvester updates. Convergence uses the relative
change in the full objective after every sweep. Random initialization is
reproducible with `random_state`; the default iteration budget is now 100.

The default direct ALS return remains `(cp_tensor, residual)`.
`return_info=True` adds a diagnostics dictionary with objective, reconstruction
error and graph-penalty histories, iteration count, status, input norm, and CG
statuses. The estimator exposes this as `diagnostics_` and also exposes
`tensor_norm_`, `objective_history_`, `n_iter_`, and `converged_`.

CG solves `A @ X + X @ B = C` using the Frobenius inner product. Real symmetric
A and B give a self-adjoint operator. Positive definiteness guarantees a unique
solution; consistent positive-semidefinite systems may also converge. Negative
operator eigenvalues are rejected. CG checks the initial residual, preserves
all caller inputs, and stops when the actual residual norm is at most
`max(atol, tol * ||C||_F)`. The unused preconditioner was removed.

Its default return remains the solution array. `return_info=True` returns
`(solution, diagnostics)`, with convergence, iteration count, residual norm, and
status: converged, max_iter, breakdown, or numerical_failure. A zero iteration
budget is supported. Both solvers warn on nonconvergence; ALS also reports an
inner_solver_failed status if a CG update fails.

## Tests and remaining limitations

Run `.venv/bin/python -m unittest discover -s tests -v` from the repository root.
Tests cover reconstruction, residuals, normalization and scale invariance,
reproducibility, objective decrease and graph smoothing, per-mode penalties,
zero tensors, higher orders, scikit-learn cloning, validation, CG agreement with
SciPy, singular systems, exact initial guesses, and iteration exhaustion.

The problem remains nonconvex and ALS does not guarantee a global optimum.
Laplacian nullspaces and unpenalized modes can permit factor-scale degeneracies;
input normalization does not remove those ambiguities. Additional constraints
or coercive factor penalties would change the agreed objective and have not
been introduced. Matrix validation uses dense eigenvalue calculations, so
large sparse graphs need a separate implementation.

Graph construction, thresholding, and an installable package/public API remain
future work. The old note identifying `Z = R` as a CG aliasing defect was
incorrect: that assignment is valid for the unpreconditioned recurrence.
