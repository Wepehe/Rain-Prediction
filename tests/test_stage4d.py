import numpy as np

from ontario_nowcast.training.stage4d_tensors import _derive


def test_stage4d_projection_derivatives_and_shear() -> None:
    spacing = 2000.0
    y, x = np.mgrid[:8, :9]
    projected_x = x.astype(np.float32) * spacing
    projected_y = y.astype(np.float32) * spacing
    # du/dx=2e-5, dv/dy=-3e-5 => convergence=+1e-5.
    u850 = 2e-5 * projected_x + 4e-5 * projected_y
    v850 = 5e-5 * projected_x - 3e-5 * projected_y
    native = {
        "UGRD_10m": np.ones_like(u850),
        "VGRD_10m": np.ones_like(u850) * 2,
        "UGRD_850hPa": u850,
        "VGRD_850hPa": v850,
        "UGRD_500hPa": u850 + 3,
        "VGRD_500hPa": v850 + 4,
        "VVEL_700hPa": np.zeros_like(u850),
    }
    result = _derive(native, projected_x, projected_y)
    np.testing.assert_allclose(result[5], 1e-5, atol=1e-9)
    # dv/dx-du/dy = 5e-5-4e-5 = 1e-5.
    np.testing.assert_allclose(result[6], 1e-5, atol=1e-9)
    np.testing.assert_allclose(result[7], 3, atol=1e-6)
    np.testing.assert_allclose(result[8], 4, atol=1e-6)
    np.testing.assert_allclose(result[9], 5, atol=1e-6)
