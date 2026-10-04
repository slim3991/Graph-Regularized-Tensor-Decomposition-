import unittest
import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import solve_sylvester
import tensorly as tl

from src.grcp import global_cg_sylvester, graph_regularized_als


def path_laplacian(size):
    adjacency = np.diag(np.ones(size - 1), 1)
    adjacency += adjacency.T
    return np.diag(adjacency.sum(axis=1)) - adjacency


class TestCG(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(4, 4))
        b = rng.normal(size=(3, 3))
        self.A = a.T @ a + np.eye(4)
        self.B = b.T @ b + np.eye(3)
        self.C = rng.normal(size=(4, 3))

    def test_agrees_with_scipy(self):
        result = global_cg_sylvester(self.A, self.B, self.C, tol=1e-10)
        np.testing.assert_allclose(
            result, solve_sylvester(self.A, self.B, self.C), atol=1e-9
        )
        self.assertLessEqual(
            np.linalg.norm(self.C - self.A @ result - result @ self.B),
            1e-10 * np.linalg.norm(self.C),
        )

    def test_sparse_matches_dense(self):
        dense = global_cg_sylvester(self.A, self.B, self.C, tol=1e-10)
        result = global_cg_sylvester(
            sparse.csr_matrix(self.A), self.B, self.C, tol=1e-10
        )
        np.testing.assert_allclose(result, dense, atol=1e-9)

    def test_nonzero_initial_guess(self):
        result = global_cg_sylvester(
            self.A, self.B, self.C, x0=np.ones_like(self.C), tol=1e-10
        )
        np.testing.assert_allclose(
            result, solve_sylvester(self.A, self.B, self.C), atol=1e-9
        )

    def test_inputs_preserved_without_initial_guess(self):
        originals = [value.copy() for value in (self.A, self.B, self.C)]
        global_cg_sylvester(self.A, self.B, self.C)
        for value, original in zip((self.A, self.B, self.C), originals):
            np.testing.assert_array_equal(value, original)

    def test_iteration_exhaustion_warns(self):
        with self.assertWarnsRegex(UserWarning, "CG did not converge"):
            result = global_cg_sylvester(
                self.A, self.B, self.C, max_iter=1, tol=1e-12
            )
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(
            np.linalg.norm(self.C - self.A @ result - result @ self.B),
            1e-12 * np.linalg.norm(self.C),
        )

    def test_consistent_singular_system(self):
        A = path_laplacian(3)
        solution = np.array([[1.], [0.], [-1.]])
        result = global_cg_sylvester(A, np.zeros((1, 1)), A @ solution)
        np.testing.assert_allclose(A @ result, A @ solution, atol=1e-10)

    def test_initial_guess_is_not_modified(self):
        initial = np.ones_like(self.C)
        original = initial.copy()
        global_cg_sylvester(self.A, self.B, self.C, x0=initial)
        np.testing.assert_array_equal(initial, original)
        initial.setflags(write=False)
        result = global_cg_sylvester(self.A, self.B, self.C, x0=initial)
        np.testing.assert_allclose(
            result, solve_sylvester(self.A, self.B, self.C), atol=1e-6
        )
        self.assertFalse(np.shares_memory(result, initial))

    def test_zero_rhs_converges_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = global_cg_sylvester(self.A, self.B, np.zeros_like(self.C))
        np.testing.assert_array_equal(result, np.zeros_like(self.C))
        self.assertEqual(caught, [])

    def test_exact_initial_guess_converges_without_warning(self):
        solution = np.ones_like(self.C)
        C = self.A @ solution + solution @ self.B
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = global_cg_sylvester(self.A, self.B, C, x0=solution.copy())
        np.testing.assert_array_equal(result, solution)
        self.assertEqual(caught, [])


class TestALS(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.get_state()
        np.random.seed(42)
        self.X = np.einsum(
            "i,j,k->ijk", [1., 3., 1., 3.], [1., 2., 3.], [1., 2.]
        )

    def tearDown(self):
        np.random.set_state(self.random_state)

    def test_rank_one_reconstruction_and_residual(self):
        cp, residual = graph_regularized_als(
            self.X, 1, [None] * 3, n_iter=5
        )
        reconstruction = tl.cp_to_tensor(cp)
        np.testing.assert_allclose(reconstruction, self.X, atol=1e-6)
        np.testing.assert_allclose(residual, self.X - reconstruction)
        weights, factors = cp
        self.assertEqual(weights.shape, (1,))
        self.assertEqual([factor.shape for factor in factors], [(4, 1), (3, 1), (2, 1)])

    def test_each_mode_accepts_dense_and_sparse_laplacians(self):
        original = self.X.copy()
        for mode, size in enumerate(self.X.shape):
            L = 0.1 * path_laplacian(size)
            original_L = L.copy()
            outputs = []
            for laplacian in (L, sparse.csr_matrix(L)):
                with self.subTest(mode=mode, sparse=sparse.issparse(laplacian)):
                    laps = [None] * 3
                    laps[mode] = laplacian
                    np.random.seed(42)
                    cp, residual = graph_regularized_als(self.X, 1, laps, n_iter=5)
                    outputs.append(tl.cp_to_tensor(cp))
                    self.assertTrue(np.isfinite(residual).all())
                    np.testing.assert_allclose(residual, self.X - outputs[-1])
                    if sparse.issparse(laplacian):
                        np.testing.assert_array_equal(laplacian.toarray(), original_L)
            np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-9)
            np.testing.assert_array_equal(L, original_L)
        np.testing.assert_array_equal(self.X, original)

    def test_graph_penalty_smooths_reconstruction(self):
        reconstructions = []
        for strength in (0., 10.):
            np.random.seed(42)
            cp, _ = graph_regularized_als(
                self.X, 1,
                [strength * path_laplacian(4) if strength else None, None, None],
                n_iter=5,
            )
            reconstructions.append(tl.cp_to_tensor(cp))
        roughness = [np.linalg.norm(np.diff(value, axis=0)) for value in reconstructions]
        self.assertLess(roughness[1], roughness[0])

    def test_invalid_tensor_order(self):
        for shape in ((4,), (4, 3), (4, 3, 2, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "three-mode"):
                graph_regularized_als(np.ones(shape), 1, [None] * len(shape))

    def test_invalid_laplacian_count_and_shape(self):
        for laps in ([None] * 2, [None] * 4):
            with self.assertRaisesRegex(ValueError, "one Laplacian"):
                graph_regularized_als(self.X, 1, laps)
        for mode in range(3):
            laps = [None] * 3
            laps[mode] = np.eye(5)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, f"mode {mode}"):
                graph_regularized_als(self.X, 1, laps)


if __name__ == "__main__":
    unittest.main()
