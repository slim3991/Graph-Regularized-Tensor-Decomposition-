import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.GRCP import MyGRTenDecomp, global_cg_sylvester, graph_regularized_als


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.tensor = np.einsum("i,j,k->ijk", [1., 2.], [2., 3., 4.], [1., 3.])

    def test_unfitted(self):
        with self.assertRaises(NotFittedError):
            MyGRTenDecomp().transform(self.tensor)

    def test_fit_reconstruction_and_residuals(self):
        original = self.tensor.copy()
        model = MyGRTenDecomp(rank=1)
        self.assertIs(model.fit(self.tensor), model)
        np.testing.assert_allclose(model.transform(self.tensor), self.tensor, atol=1e-6)
        np.testing.assert_allclose(model.residuals(self.tensor), model.E_)
        np.testing.assert_array_equal(self.tensor, original)

    def test_fit_transform_and_integer_input(self):
        result = MyGRTenDecomp(rank=1).fit_transform(self.tensor.astype(int))
        np.testing.assert_allclose(result, self.tensor, atol=1e-6)

    def test_transform_is_protected_and_checks_shape(self):
        model = MyGRTenDecomp(rank=1).fit(self.tensor)
        model.transform(self.tensor)[:] = 0
        np.testing.assert_allclose(model.X_hat_, self.tensor, atol=1e-6)
        with self.assertRaises(ValueError):
            model.transform(np.ones((3, 3, 3)))

    def test_invalid_tensors(self):
        for tensor in [np.ones((2, 2)), np.empty((0, 2, 2)),
                       np.zeros((2, 2, 2)), np.full((2, 2, 2), np.nan),
                       np.full((2, 2, 2), np.inf), np.ones((2, 2, 2), complex),
                       np.full((2, 2, 2), "bad")]:
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype):
                with self.assertRaises(ValueError):
                    MyGRTenDecomp().fit(tensor)

    def test_invalid_parameters(self):
        for rank in [0, -1, 1.5, True]:
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                MyGRTenDecomp(rank=rank).fit(self.tensor)
        for tol in [0, -1, np.nan, np.inf, "bad", True]:
            with self.subTest(tol=tol), self.assertRaises(ValueError):
                MyGRTenDecomp(tol=tol).fit(self.tensor)

    def test_unimplemented_options_are_explicit(self):
        for options in [{"local_threshold": 0.5}, {"laplacian_parameters": {"sigma": 1.}}]:
            with self.subTest(options=options), self.assertRaises(NotImplementedError):
                MyGRTenDecomp(**options).fit(self.tensor)

    def test_laplacians_are_validated(self):
        for laps in [[None], [np.eye(3), None, None],
                     [np.array([[0., 1.], [0., 0.]]), None, None],
                     [np.full((2, 2), np.nan), None, None]]:
            with self.subTest(laps=laps), self.assertRaises(ValueError):
                MyGRTenDecomp(laps=laps).fit(self.tensor)

    def test_laplacians_reach_solver(self):
        lap = np.array([[1., -1.], [-1., 1.]])
        model = MyGRTenDecomp(rank=1, laps=[lap, None, None])
        with patch("src.GRCP.graph_regularized_als", wraps=graph_regularized_als) as solver:
            model.fit(self.tensor)
        np.testing.assert_array_equal(solver.call_args.kwargs["laps"][0], lap)
        self.assertIsNot(model.laps_[0], lap)
        self.assertTrue(np.all(np.isfinite(model.X_hat_)))

    def test_clone(self):
        model = MyGRTenDecomp(rank=1).fit(self.tensor)
        copied = clone(model)
        self.assertEqual(copied.get_params(), model.get_params())
        self.assertFalse(hasattr(copied, "X_hat_"))

    def test_cg_solves_known_sylvester_equation(self):
        A, B = np.diag([1., 2.]), np.diag([3., 4.])
        expected = np.array([[1., 2.], [3., 4.]])
        C = A @ expected + expected @ B
        np.testing.assert_allclose(global_cg_sylvester(A, B, C), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
