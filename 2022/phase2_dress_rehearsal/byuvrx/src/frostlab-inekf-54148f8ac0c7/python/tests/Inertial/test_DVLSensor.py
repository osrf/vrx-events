import numpy as np
from numpy.testing import assert_allclose
from inekf import SE3, DVLSensor
import pytest

def test_processZ():
    state = SE3[2,6]()
    # append IMU values to the end of it
    z = np.array([1,2,3,0,0,0])

    ds = DVLSensor()
    z_full = ds.processZ(z, state)

    expected = np.array([1,2,3,-1,0])
    assert_allclose(expected, z_full)