import unittest

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.grcp import MyGRTenDecomp


class TestWrapper(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(1.0, 25.0).reshape(3, 4, 2)
        self.model = MyGRTenDecomp([None, None, None], rank=1, n_iter=2)

    def test_unfitted_state_and_clone(self):
        for operation in (
            lambda: check_is_fitted(self.model),
            lambda: self.model.transform(self.X),
            lambda: self.model.residuals(self.X),
        ):
            with self.assertRaises(NotFittedError):
                operation()
        cloned = clone(self.model)
        self.assertEqual(cloned.get_params(), self.model.get_params())
        self.assertIs(cloned.set_params(n_iter=3), cloned)
        self.assertEqual(cloned.n_iter, 3)

    def test_fit_transform_and_output_isolation(self):
        original = self.X.copy()
        reconstruction = self.model.fit_transform(self.X, y=np.ones(3))
        check_is_fitted(self.model)
        np.testing.assert_allclose(reconstruction, self.model.X_hat_)
        np.testing.assert_allclose(self.model.residuals(self.X), self.model.E_)
        np.testing.assert_array_equal(self.X, original)
        reconstruction[:] = 0
        self.assertTrue(np.any(self.model.X_hat_ != 0))
        self.assertIs(self.model.fit(self.X), self.model)

    def test_validation_and_refit(self):
        self.model.fit(self.X.tolist())
        with self.assertRaises(ValueError):
            self.model.transform(np.ones((2, 4, 2)))
        with self.assertRaises(ValueError):
            self.model.fit(np.full(self.X.shape, np.nan))
        self.model.fit(np.ones((2, 3, 4)))
        self.assertEqual(self.model.tensor_shape_, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
