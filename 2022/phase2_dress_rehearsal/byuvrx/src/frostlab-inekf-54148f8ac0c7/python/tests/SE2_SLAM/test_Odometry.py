import numpy as np
from numpy.testing import assert_allclose
from inekf import SE2, OdometryProcess, ERROR
import pytest

def test_f():
    state = SE2()
    u = SE2(.1, 1, 2)

    op = OdometryProcess()

    assert_allclose(op.f(u, 1, state).State, u.State)

def test_MakePhi():
    state = SE2()
    u = SE2(.1, 1, 2)

    op = OdometryProcess()

    phi = op.MakePhi(u, 1, state, ERROR.RIGHT)
    assert_allclose(phi, np.eye(3))

    phi = op.MakePhi(u, 1, state, ERROR.LEFT)
    assert_allclose(phi, u.inverse().Ad())