# GRCP cleanup and remaining work

The source file is `src/GRCP.py` (case-sensitive). The estimator currently
supports finite, nonzero, three-dimensional real NumPy tensors and optional
precomputed symmetric matrices, one per tensor mode. `transform(X)` returns
the training reconstruction and requires the training shape. It does not fit
a new tensor. `residuals(X)` subtracts that reconstruction from the supplied X.

## Changes made

- Corrected the NumPy type alias and scikit-learn fitted-state import.
- Removed the unused time import and premature learned attributes.
- Put TransformerMixin first in the inheritance list for scikit-learn mixin behavior.
- Validate tensor shape, numeric values, rank, tolerance, and Laplacian dimensions
  and symmetry before calling the solver; convert integer inputs to floating point.
- Forward precomputed Laplacians and retain the CP tensor, reconstruction,
  residual, Laplacians, and training shape as fitted attributes.
- Return a copy of the reconstruction so callers cannot modify fitted state.
- Raise NotImplementedError for requested graph construction or thresholding,
  which were previously silently ignored. Empty graph parameters remain accepted.
- Declare scikit-learn and update the dependency lockfile.
- Add 11 unittest functionality tests, including a real decomposition and a
  known Sylvester equation solution. Neither solver function was changed.

## Remaining changes needed for the intended library

The supported estimator path works in the tested cases. The following work is
still needed for the broader intended functionality:

1. Define and implement graph construction from `laplacian_parameters`, including
   graph weights and regularization strength. Implement or remove
   `local_threshold` and define the meaning of the former `threshold_` attribute.
2. Decide whether to support arbitrary tensor order. ALS currently hardcodes
   three modes and two other-mode Gram matrices. General support requires
   products across all other modes; the wrapper currently rejects other orders.
3. Harden direct solver calls: validate shapes and parameters, handle zero-norm
   tensors in ALS, and handle `max_iter=0` in CG (its warning references an
   undefined loop variable).
4. Correct CG residual aliasing: `Z = R` followed by in-place `R` updates changes
   Z before the next inner product. Copy the residual or restructure the update.
   Also define ownership of `x0`, which is currently modified in place.
5. Handle an initially converged CG solution, numerical breakdown, and exhausted
   iterations distinctly. Validate the positive-definiteness assumptions for CG
   and decide whether to activate the computed diagonal preconditioner.
6. Define ALS convergence for the graph-regularized objective, rather than only
   reconstruction error, expose iteration/status diagnostics, and implement or
   remove unused verbosity options. Add numerical tests for these cases when
   changing the solvers; the current smoke tests do not establish robustness.
7. Add reproducible random initialization and define whether the estimator will
   ever transform new observations rather than return its training reconstruction.
8. Define an installable package layout and public API; imports currently use
   `src.GRCP` from the repository root, and `main.py` is still a placeholder.

Run tests from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -v
```
