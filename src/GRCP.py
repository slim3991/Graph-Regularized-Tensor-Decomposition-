import warnings
from numbers import Real
import numpy.typing as npt
import numpy as np
import tensorly as tl
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao


type Tensor = npt.NDArray


class MyGRTenDecomp(TransformerMixin, BaseEstimator):
    """Reconstruct a NumPy tensor of order two or higher with CP decomposition.

    ``transform`` returns the fitted reconstruction for tensors of the same
    shape; it does not decompose new observations. Supply precomputed graph
    Laplacians through ``laps``. Graph construction and thresholding are not
    implemented yet.
    """

    def __init__(
        self,
        laplacian_parameters: Optional[Dict[str, float | str]] = None,
        rank: int = 5,
        local_threshold: Optional[float] = None,
        tol=1e-6,
        laps: Optional[List[Optional[Tensor]]] = None,
        regularization_strengths=None,
        n_iter=100,
        random_state=None,
    ):
        self.rank = rank
        self.laplacian_parameters = laplacian_parameters
        self.local_threshold = local_threshold
        self.tol = tol
        self.laps = laps
        self.regularization_strengths = regularization_strengths
        self.n_iter = n_iter
        self.random_state = random_state

    @staticmethod
    def name():
        return "GR-CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        X = self._validate_tensor(X)
        if self.laplacian_parameters:
            raise NotImplementedError(
                "Graph construction is not implemented; supply precomputed laps"
            )
        if self.local_threshold is not None:
            raise NotImplementedError("local_threshold is not implemented")
        laps, _ = _graphs(X.shape, self.laps, self.regularization_strengths)
        factors, residual, info = graph_regularized_als(
            X,
            self.rank,
            tol=self.tol,
            laps=laps,
            regularization_strengths=self.regularization_strengths,
            n_iter=self.n_iter,
            random_state=self.random_state,
            return_info=True,
        )
        reconstruction = tl.cp_to_tensor(factors)
        self.cp_tensor_ = factors
        self.X_hat_ = reconstruction
        self.E_ = residual
        self.laps_ = laps
        self.tensor_shape_ = X.shape
        self.diagnostics_ = info
        self.tensor_norm_ = info["tensor_norm"]
        self.objective_history_ = info["objective_history"]
        self.n_iter_ = info["n_iter"]
        self.converged_ = info["converged"]

        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self, ["X_hat_"])
        X = self._validate_tensor(X)
        if X.shape != self.tensor_shape_:
            raise ValueError("X must have the same shape as the fitted tensor")
        return self.X_hat_.copy()

    def residuals(self, X: Tensor) -> Tensor:
        X_hat = self.transform(X)
        return X - X_hat

    @staticmethod
    def _validate_tensor(X: Tensor) -> Tensor:
        X = _real_array(X, "X")
        if X.ndim < 2 or any(size == 0 for size in X.shape):
            raise ValueError("X must have at least two nonempty dimensions")
        return X


def _positive_integer(value, name, allow_zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < (0 if allow_zero else 1)
    ):
        raise ValueError(
            f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer"
        )


def _real_parameter(value, name, allow_zero=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        raise ValueError(
            f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number"
        )


def _real_array(value, name):
    result = np.asarray(value)
    if (
        not np.issubdtype(result.dtype, np.number)
        or np.iscomplexobj(result)
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must contain finite real numbers")
    return result.astype(float, copy=True)


def _symmetric_matrix(value, name):
    result = _real_array(value, name)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or not result.size:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return (result + result.T) / 2


def _graphs(shape, laps, regularization_strengths):
    matrices = [None] * len(shape) if laps is None else list(laps)
    strengths = (
        np.ones(len(shape))
        if regularization_strengths is None
        else _real_array(regularization_strengths, "regularization_strengths")
    )
    if len(matrices) != len(shape) or strengths.shape != (len(shape),):
        raise ValueError(
            "Supply one Laplacian and regularization strength per tensor mode"
        )
    if np.any(strengths < 0):
        raise ValueError("regularization_strengths must be nonnegative")
    for mode, lap in enumerate(matrices):
        if lap is not None:
            lap = _symmetric_matrix(lap, "Laplacian")
            if lap.shape != (shape[mode], shape[mode]):
                raise ValueError("Laplacian shape must match its tensor mode")
            if np.linalg.eigvalsh(lap)[0] < -1e-10 * max(1.0, np.linalg.norm(lap, 2)):
                raise ValueError("Laplacians must be positive semidefinite")
            matrices[mode] = lap
    return matrices, strengths


def graph_regularized_als(
    tensor: npt.NDArray,
    rank: int,
    n_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    laps: List[npt.NDArray] | List[None] | None = None,
    *,
    regularization_strengths=None,
    random_state=None,
    return_info=False,
):
    """Fit CP on X / ||X|| with one graph penalty per mode.

    Objective: 0.5 * ||X_normalized - CP(U)||_F**2 +
    0.5 * sum(lambda[m] * trace(U[m].T @ L[m] @ U[m])).
    Factors are not rescaled during optimization: doing so changes the penalty.
    Returned CP weights restore the original input scale. Optional diagnostics
    describe the normalized objective. None Laplacians disable that mode's penalty.
    """
    tensor = _real_array(tensor, "tensor")
    if tensor.ndim < 2 or any(size == 0 for size in tensor.shape):
        raise ValueError("tensor must have at least two nonempty dimensions")
    _positive_integer(rank, "rank")
    _positive_integer(n_iter, "n_iter")
    _real_parameter(tol, "tol")
    laps, strengths = _graphs(tensor.shape, laps, regularization_strengths)
    scale = float(np.linalg.norm(tensor))
    if not np.isfinite(scale):
        raise ValueError("tensor norm must be finite")
    if scale == 0:
        cp = (np.zeros(rank), [np.zeros((size, rank)) for size in tensor.shape])
        info = dict(
            converged=True,
            status="converged",
            n_iter=0,
            objective_history=[0.0],
            reconstruction_error_history=[0.0],
            graph_penalty_history=[0.0],
            tensor_norm=0.0,
            cg_statuses=[],
        )
        result = (cp, tensor.copy())
        return (*result, info) if return_info else result
    normalized = tensor / scale
    rng = np.random.default_rng(random_state)
    factors = [
        rng.standard_normal((size, rank)) / np.sqrt(size) for size in tensor.shape
    ]
    weights = np.ones(rank)

    def objective():
        error = 0.5 * float(
            np.linalg.norm(normalized - tl.cp_to_tensor((weights, factors))) ** 2
        )
        penalty = 0.5 * sum(
            strengths[m] * float(np.sum(factors[m] * (lap @ factors[m])))
            for m, lap in enumerate(laps)
            if lap is not None
        )
        return error + penalty, error, penalty

    total, error, penalty = objective()
    history, errors, penalties, cg_statuses = [total], [error], [penalty], []
    converged = False
    status = "max_iter"
    for iteration in range(1, n_iter + 1):
        inner_failed = False
        for mode in range(tensor.ndim):
            gram = np.ones((rank, rank))
            for other in range(tensor.ndim):
                if other != mode:
                    gram *= factors[other].T @ factors[other]
            rhs = unfolding_dot_khatri_rao(normalized, (weights, factors), mode)
            if laps[mode] is None or strengths[mode] == 0:
                factors[mode] = np.linalg.lstsq(gram, rhs.T, rcond=None)[0].T
            else:
                factors[mode], cg_info = global_cg_sylvester(
                    strengths[mode] * laps[mode],
                    gram,
                    rhs,
                    x0=factors[mode],
                    tol=min(tol * 0.1, 1e-8),
                    atol=1e-12,
                    max_iter=2000,
                    return_info=True,
                )
                cg_statuses.append(dict(iteration=iteration, mode=mode, **cg_info))
                inner_failed |= not cg_info["converged"]
        total, error, penalty = objective()
        history.append(total)
        errors.append(error)
        penalties.append(penalty)
        if verbose:
            print(f"ALS iteration {iteration}: objective={total:.8g}")
        if not np.isfinite(total):
            status = "numerical_failure"
            break
        if inner_failed:
            status = "inner_solver_failed"
            break
        if abs(history[-2] - total) <= tol * max(abs(history[-2]), np.finfo(float).eps):
            converged, status = True, "converged"
            break
    if not converged:
        warnings.warn(
            f"ALS did not converge: {status} after {iteration} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    cp = (weights * scale, factors)
    info = dict(
        converged=converged,
        status=status,
        n_iter=iteration,
        objective_history=history,
        reconstruction_error_history=errors,
        graph_penalty_history=penalties,
        tensor_norm=scale,
        cg_statuses=cg_statuses,
    )
    result = (cp, tensor - tl.cp_to_tensor(cp))
    return (*result, info) if return_info else result


def global_cg_sylvester(
    A,
    B,
    C,
    x0=None,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
    *,
    atol=1e-12,
    return_info=False,
):
    """Solve A X + X B = C with unpreconditioned Frobenius-inner-product CG.

    A and B must be real symmetric and their summed operator positive
    semidefinite; positive definiteness guarantees a unique solution. Consistent
    singular systems can converge, otherwise breakdown is reported. Stop when
    ||C - A X - X B|| <= max(atol, tol * ||C||). Inputs are never mutated.
    """
    A, B = _symmetric_matrix(A, "A"), _symmetric_matrix(B, "B")
    C = _real_array(C, "C")
    if C.shape != (A.shape[0], B.shape[0]):
        raise ValueError("C shape must match A and B")
    _positive_integer(max_iter, "max_iter", allow_zero=True)
    _real_parameter(tol, "tol", allow_zero=True)
    _real_parameter(atol, "atol", allow_zero=True)
    minimum = np.linalg.eigvalsh(A)[0] + np.linalg.eigvalsh(B)[0]
    operator_scale = np.linalg.norm(A, 2) + np.linalg.norm(B, 2)
    if minimum < -1e-12 * operator_scale:
        raise ValueError("The Sylvester operator must be positive semidefinite")
    X = np.zeros_like(C) if x0 is None else _real_array(x0, "x0")
    if X.shape != C.shape:
        raise ValueError("x0 must have the same shape as C")
    threshold = max(atol, tol * float(np.linalg.norm(C)))
    R = C - (A @ X + X @ B)
    P = R.copy()
    rr = float(np.vdot(R, R).real)
    status = "converged" if np.linalg.norm(R) <= threshold else "max_iter"
    iterations = 0
    if status != "converged":
        for iterations in range(1, max_iter + 1):
            W = A @ P + P @ B
            curvature = float(np.vdot(P, W).real)
            if not np.isfinite(curvature) or not np.isfinite(rr):
                status = "numerical_failure"
                break
            if curvature <= np.finfo(float).eps * np.linalg.norm(P) * np.linalg.norm(W):
                status = "breakdown"
                break
            alpha = rr / curvature
            X += alpha * P
            R -= alpha * W
            if np.linalg.norm(R) <= threshold:
                # Confirm convergence with the actual equation residual.
                R = C - (A @ X + X @ B)
                if np.linalg.norm(R) <= threshold:
                    status = "converged"
                    break
                P = R.copy()
                rr = float(np.vdot(R, R).real)
                continue
            rr_new = float(np.vdot(R, R).real)
            P = R + (rr_new / rr) * P
            rr = rr_new
            if verbose:
                print(f"CG iteration {iterations}: residual={np.linalg.norm(R):.8g}")
    residual_norm = float(np.linalg.norm(C - (A @ X + X @ B)))
    info = dict(
        converged=status == "converged",
        status=status,
        n_iter=iterations,
        residual_norm=residual_norm,
    )
    if not info["converged"]:
        warnings.warn(
            f"CG did not converge: {status} after {iterations} iterations",
            RuntimeWarning,
            stacklevel=2,
        )
    return (X, info) if return_info else X
