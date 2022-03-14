import numpy as np
from numpy.testing import assert_allclose
from inekf import SE2, GenericMeasureModel, ERROR
import pytest

Group = SE2[2,1]

def test_bConstructor():
    b = np.array([0,0,0,1])
    M = np.eye(2) 
    H = np.zeros((2,6))
    H[0:2,3:5] = np.eye(2)

    l = GenericMeasureModel[Group](b, M, ERROR.LEFT)
    assert_allclose(H, l.H)

    r = GenericMeasureModel[Group](b, M, ERROR.RIGHT)
    assert_allclose(-H, r.H)

    b[0] = 1
    with pytest.raises(Exception):
        l = GenericMeasureModel[Group](b, M, ERROR.LEFT)

def test_processZ():
    b = np.array([0,0,0,1])
    M = np.eye(2) 
    state = Group()
    S = GenericMeasureModel[Group](b, M, ERROR.LEFT)

    z = np.array([2,2])
    expected = np.array([2,2,0,1])

    assert_allclose(expected, S.processZ(z, state))