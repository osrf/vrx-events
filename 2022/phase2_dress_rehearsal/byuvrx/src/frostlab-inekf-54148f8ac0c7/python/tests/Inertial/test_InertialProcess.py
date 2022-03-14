import numpy as np
from numpy.testing import assert_allclose
from inekf import SE3, SO3, DVLSensor, InertialProcess
import pytest

def test_f():
    state = SE3[2,6]()
    u = np.array([1,1,1,1,1,1+9.81])

    ip = InertialProcess()
    result = ip.f(u, 1, state)

    assert_allclose(result.R().State, SO3.Exp(u[:3]).State)
    assert_allclose(result[0], np.ones(3))
    assert_allclose(result[1], np.ones(3)/2)
    