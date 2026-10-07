import numpy as np

from ontario_nowcast.training.stage6_prepare import blend


def test_frozen_hybrid_formula():
    a = np.asarray([0.0, 1.0, 0.2, 0.8])
    h = np.asarray([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(blend(a, h), np.asarray([0.0, 0.5, 0.6, 0.9]))
