import warnings
import time
import numpy.typing as npt
import numpy as np
from scipy import sparse
import tensorly as tl
from typing import Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao

type Tensor = npt.NDArray[np.floating]


class MyGRTenDecomp(TransformerMixin, BaseEstimator):
    """Decompose one tensor and return its fitted reconstruction.

    ``transform`` requires the fitted tensor's shape; it does not fit new
    observations. ``y`` is accepted for sklearn compatibility and ignored.
    """

    def __init__(
        self,
        laplacians: Sequence[npt.NDArray | sparse.spmatrix | None],
        *,
        rank: int = 5,
        tol=1e-6,
        n_iter=20,
    ):
        self.rank = rank
        self.laplacians = laplacians
        self.tol = tol
        self.n_iter = n_iter

    @staticmethod
    def name():
        return "GR-CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        """Fit the decomposition and return self."""
        X = check_array(X, allow_nd=True, dtype=np.float64)

        factors, residual = graph_regularized_als(
            X,
            self.rank,
            laplacians=self.laplacians,
            tol=self.tol,
            n_iter=self.n_iter,
        )

        self.X_hat_ = tl.cp_to_tensor(factors)
        self.cp_tensor_ = factors
        self.E_ = residual
        self.laps_ = tuple(self.laplacians)
        self.tensor_shape_ = X.shape

        return self

    def transform(self, X: Tensor) -> Tensor:
        """Return a copy of the fitted reconstruction for a matching shape."""
        check_is_fitted(self, ["X_hat_", "tensor_shape_"])
        X = check_array(X, allow_nd=True, dtype=np.float64)
        if X.shape != self.tensor_shape_:
            raise ValueError("X must have the same shape as the fitted tensor.")
        return self.X_hat_.copy()

    def residuals(self, X: Tensor) -> Tensor:
        X_hat = self.transform(X)
        return X - X_hat


def graph_regularized_als(
    tensor,
    rank,
    laplacians,
    n_iter=20,
    tol=1e-6,
    verbose=False,
):
    """Fit a three-mode CP tensor using supplied, optionally scaled Laplacians.

    Supply one dense or sparse matrix per mode, or None to disable that
    mode's graph penalty. Each matrix must match its mode's size.
    CP weights stay at one and factors are not normalized during optimization.
    Convergence uses relative change in the full objective: half the squared
    reconstruction error plus half the sum of graph quadratic penalties.
    """
    if tensor.ndim != 3:
        raise ValueError("Expected a three-mode tensor.")
    if len(laplacians) != tensor.ndim:
        raise ValueError("Supply one Laplacian or None per tensor mode.")
    for mode, laplacian in enumerate(laplacians):
        if laplacian is not None and laplacian.shape != (tensor.shape[mode],) * 2:
            raise ValueError(f"Laplacian for mode {mode} must match its mode's size.")

    # Initial CP Decomposition
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, tol=tol, init="random"
    )
    # Absorb component amplitudes once; subsequent updates keep unit weights.
    if weights is not None:
        factors[0] = factors[0] * weights[None, :]
    weights = np.ones(rank, dtype=factors[0].dtype)

    def objective():
        residual = tensor - tl.cp_to_tensor((weights, factors))
        graph_penalty = sum(
            np.sum(factor * (laplacian @ factor))
            for factor, laplacian in zip(factors, laplacians)
            if laplacian is not None
        )
        return 0.5 * (np.sum(residual**2) + graph_penalty)

    old_objective = objective()

    for i in range(n_iter):
        for mode in range(3):
            # --- 1. Setup components ---
            idx = [m for m in range(3) if m != mode]
            G1 = np.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = np.dot(factors[idx[1]].T, factors[idx[1]])
            S = G1 * G2

            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode=mode)

            if laplacians[mode] is None:
                factors[mode] = np.linalg.lstsq(S, mttkrp.T, rcond=None)[0].T
            else:
                X = factors[mode]
                X = global_cg_sylvester(
                    A=laplacians[mode],
                    B=S,
                    C=mttkrp,
                    x0=X,
                    verbose=False,
                    tol=tol,
                    max_iter=2000,
                )
                factors[mode] = X

        current_objective = objective()
        delta = abs(current_objective - old_objective)
        scale = max(abs(old_objective), np.finfo(float).eps)
        if verbose:
            print(f"Iter {i}, Objective: {current_objective:.6g}")
        if delta <= tol * scale:
            break
        old_objective = current_objective

    return (weights, factors), (tensor - tl.cp_to_tensor((weights, factors)))


def global_cg_sylvester(A, B, C, x0=None, max_iter=1000, tol=1e-6, verbose=False):

    t0 = time.perf_counter()
    if x0 is None:
        X = np.zeros_like(C)
        R = C.copy()
    else:
        X = x0.copy()
        R = C - (A @ X + X @ B)

    residual_tolerance_squared = (tol**2) * np.vdot(C, C).real
    if np.vdot(R, R).real <= residual_tolerance_squared:
        if verbose:
            print("Converged in 0 iterations")
        return X

    Z = R
    P = Z.copy()

    rz_inner = np.vdot(R, Z).real
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    for k in range(max_iter):
        W = A @ P + P @ B

        denom_cg = np.vdot(P, W).real
        if denom_cg <= 1e-16:
            if verbose:
                print(f"Breakdown at iter {k}")
            break

        alpha = rz_inner / denom_cg

        X += alpha * P
        R -= alpha * W

        # Check convergence relative to the right-hand side.
        if np.vdot(R, R).real <= residual_tolerance_squared:
            if verbose:
                t_loop = time.perf_counter() - t1
                print("=" * 30)
                print(f"Setup: {t_setup:.4f}s | Loop ({k} iters): {t_loop:.4f}s")
                print(f"Converged in {k+1} iterations")
                print("=" * 30 + "\n")
            return X

        Z = R

        rz_new = np.vdot(R, Z).real
        beta = rz_new / rz_inner

        P = Z + beta * P
        rz_inner = rz_new

    if verbose:
        t_loop = time.perf_counter() - t1
        print(f"Setup: {t_setup:.4f}s | Loop ({k} iters): {t_loop:.4f}s")
    warnings.warn(f"CG did not converge, finished at iteration: {k}")
    return X
