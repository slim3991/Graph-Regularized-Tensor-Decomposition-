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
    """Reconstruct a three-dimensional NumPy tensor with CP decomposition.

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
    ):
        self.rank = rank
        self.laplacian_parameters = laplacian_parameters
        self.local_threshold = local_threshold
        self.tol = tol
        self.laps = laps

    @staticmethod
    def name():
        return "GR-CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        X = self._validate_tensor(X)
        if (
            isinstance(self.rank, (bool, np.bool_))
            or not isinstance(self.rank, (int, np.integer))
            or self.rank < 1
        ):
            raise ValueError("rank must be a positive integer")
        if (
            isinstance(self.tol, (bool, np.bool_))
            or not isinstance(self.tol, Real)
            or not np.isfinite(self.tol)
            or self.tol <= 0
        ):
            raise ValueError("tol must be a finite positive number")
        if self.laplacian_parameters:
            raise NotImplementedError(
                "Graph construction is not implemented; supply precomputed laps"
            )
        if self.local_threshold is not None:
            raise NotImplementedError("local_threshold is not implemented")
        laps = [None] * X.ndim if self.laps is None else list(self.laps)
        if len(laps) != X.ndim:
            raise ValueError("laps must contain one entry per tensor mode")
        for mode, lap in enumerate(laps):
            if lap is not None:
                lap = np.asarray(lap)
                if lap.shape != (X.shape[mode], X.shape[mode]):
                    raise ValueError(
                        "Each Laplacian must be square and match its tensor mode"
                    )
                if (
                    not np.issubdtype(lap.dtype, np.number)
                    or np.iscomplexobj(lap)
                    or not np.all(np.isfinite(lap))
                ):
                    raise ValueError("Laplacians must contain finite real numbers")
                if not np.allclose(lap, lap.T):
                    raise ValueError("Laplacians must be symmetric")
                laps[mode] = lap.astype(float, copy=True)

        factors, residual = graph_regularized_als(
            X,
            self.rank,
            tol=self.tol,
            laps=laps,
        )
        reconstruction = tl.cp_to_tensor(factors)
        self.cp_tensor_ = factors
        self.X_hat_ = reconstruction
        self.E_ = residual
        self.laps_ = laps
        self.tensor_shape_ = X.shape

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
        X = np.asarray(X)
        if X.ndim != 3 or any(size == 0 for size in X.shape):
            raise ValueError("X must be a nonempty three-dimensional tensor")
        if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
            raise ValueError("X must contain real numbers")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        X = X.astype(float, copy=True)
        if np.linalg.norm(X) == 0:
            raise ValueError("X must have nonzero norm")
        return X


def graph_regularized_als(
    tensor: npt.NDArray,
    rank: int,
    n_iter: int = 20,
    tol: float = 1e-6,
    verbose: bool = False,
    laps: List[npt.NDArray] | List[None] | None = None,
):
    dims = len(tensor.shape)
    if laps is None:
        laps = [None] * dims

    # Initial CP Decomposition
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, tol=tol, init="random"
    )

    M = tensor.copy()
    old_err = 1e10

    for i in range(n_iter):
        for mode in range(3):
            idx = [m for m in range(3) if m != mode]
            G1 = np.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = np.dot(factors[idx[1]].T, factors[idx[1]])
            S = G1 * G2 * (weights[:, None] * weights[None, :])

            mttkrp = unfolding_dot_khatri_rao(M, (weights, factors), mode=mode)

            if laps[mode] is None:
                factors[mode] = np.linalg.solve(S + 1e-8 * np.eye(rank), mttkrp.T).T
            else:
                X = factors[mode]
                X = global_cg_sylvester(
                    A=laps[mode],
                    B=S,
                    C=mttkrp,
                    x0=X,
                    verbose=False,
                    tol=tol,
                    max_iter=2000,
                )
                factors[mode] = X

        for r in range(rank):
            norm = 1.0
            for mode in range(3):
                col_norm = np.linalg.norm(factors[mode][:, r]) + 1e-12
                factors[mode][:, r] /= col_norm
                norm *= col_norm
            weights[r] *= norm

        if i == 0 or i % 4 == 0:

            X_tensor = tl.cp_to_tensor((weights, factors))
            res = tensor - X_tensor

            err = np.linalg.norm(res) / tl.norm(M)
            delta = np.abs(err - old_err)

            if delta < tol:
                break
            old_err = err

    return (weights, factors), (tensor - tl.cp_to_tensor((weights, factors)))


def global_cg_sylvester(A, B, C, x0=None, max_iter=1000, tol=1e-6, verbose=False):

    dA = A.diagonal()
    dB = B.diagonal()

    # Avoid division by zero
    denom = dA[:, None] + dB[None, :]
    denom[denom == 0] = 1e-12

    if x0 is None:
        X = np.zeros_like(C)
        R = C.copy()
    else:
        X = x0
        R = C - (A @ X + X @ B)

    Z = R  # / denom
    P = Z.copy()

    rz_inner = np.vdot(R, Z).real
    for k in range(max_iter):
        W = A @ P + P @ B

        denom_cg = np.vdot(P, W).real
        if denom_cg <= 1e-16:
            break

        alpha = rz_inner / denom_cg

        X += alpha * P
        R -= alpha * W

        # Check convergence (relative to initial residual)
        if np.vdot(R, R).real <= (tol**2) * np.vdot(C, C).real:
            return X

        Z = R  # / denom

        rz_new = np.vdot(R, Z).real
        beta = rz_new / rz_inner

        P = Z + beta * P
        rz_inner = rz_new

        # if verbose and k % 10 == 0:
        #     res = np.sqrt(np.vdot(R, R).real)
        #     print(f"iter {k}, residual {res:.2e}")
    warnings.warn(f"CG did not converge, finished at iteration: {k}")
    return X
