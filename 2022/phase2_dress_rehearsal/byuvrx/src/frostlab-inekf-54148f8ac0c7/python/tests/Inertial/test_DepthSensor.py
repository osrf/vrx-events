import numpy as np
from numpy.testing import assert_allclose
from inekf import SE3, DepthSensor
import pytest

def test_processZ():
    state = SE3[2,6]()
    z = np.array([5])

    ds = DepthSensor()
    z_full = ds.processZ(z, state)

    expected = np.array([0,0,5,0,1])
    assert_allclose(expected, z_full)