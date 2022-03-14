import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from numpy.linalg import inv
from inekf import SO2
import pytest

def test_BaseConstructor1():
    mtx = np.eye(2)
    sigma = np.eye(3)
    A = np.ones(2)

    x = SO2[2](mtx, sigma, A)

    assert_allclose(mtx, x.State)
    assert_allclose(sigma, x.Cov)
    assert_allclose(A, x.Aug)

def test_BaseConstructor2():
    mtx = np.eye(2)
    sigma = np.eye(3)
    A = np.ones(2)

    with pytest.raises(Exception):
        SO2["D"](mtx, sigma)

    SO2["D"](mtx, sigma, A)
    SO2["D"]()

def test_ThetaConstructor():
    x = SO2(np.pi/4)
    v = 1 / np.sqrt(2)
    r = np.array([[v, -v],[v, v]])

    assert_allclose(r, x.State)

def test_TangentConstructor():
    x = np.arange(3)

    state = SO2["D"](x)

    assert_allclose(x[1:], state.Aug)
    assert_allclose(state.State, np.eye(2))

def test_AddAug():
    x = SO2["D"]()

    x.addAug(2)
    assert x.Aug[-1] == 2

    with pytest.raises(Exception):
        y = SO2()
        y.addAug(2)

def test_Inverse():
    x = SO2()
    assert_allclose(inv(x.State), x.inverse().State)

def test_Exp():
    x = np.arange(1,4)

    ours = SO2["D"].Exp(x)
    theirs = expm( SO2["D"].Wedge(x) )

    assert_allclose(theirs, ours.State)
    assert_allclose(x[-2:], ours.Aug)
    with pytest.raises(Exception):
        SO2[3].Exp(x)

def test_Log():
    xi = np.arange(3)
    x = SO2["D"](xi)

    assert_allclose(xi, x.log())

def test_Wedge():
    x = np.arange(1,4)
    ours = SO2["D"].Wedge(x)
    theirs =  np.array([[0, -1],
                        [1, 0]])
    
    assert_allclose(theirs, ours)

    with pytest.raises(Exception):
        SO2[3].Wedge(x)

def test_Adjoint():
    assert_allclose(np.ones(1), SO2().Ad())
    assert_allclose(np.eye(3), SO2[2]().Ad())