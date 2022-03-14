import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from numpy.linalg import inv
from inekf import SO3
import pytest

def test_BaseConstructor1():
    mtx = np.eye(3)
    sigma = np.eye(5)
    A = np.ones(2)

    x = SO3[2](mtx, sigma, A)

    assert_allclose(mtx, x.State)
    assert_allclose(sigma, x.Cov)
    assert_allclose(A, x.Aug)

def test_BaseConstructor2():
    mtx = np.eye(3)
    sigma = np.eye(5)
    A = np.ones(2)

    with pytest.raises(Exception):
        SO3["D"](mtx, sigma)

    SO3["D"](mtx, sigma, A)
    SO3["D"]()

def test_ThetaConstructor():
    x = SO3(np.pi/4, 0, 0)
    v = 1 / np.sqrt(2)
    r = np.array([[1, 0, 0],
                [0, v, -v],
                [0, v, v]])

    assert_allclose(r, x.State)

def test_TangentConstructor():
    x = np.array([0,0,0,4,5])

    state = SO3["D"](x)

    assert_allclose(x[-2:], state.Aug)
    assert_allclose(state.State, np.eye(3))

def test_AddAug():
    x = SO3["D"]()

    x.addAug(2)
    assert x.Aug[-1] == 2

    with pytest.raises(Exception):
        y = SO3()
        y.addAug(2)

def test_Inverse():
    x = SO3(np.pi/4, np.pi/4, np.pi/4)
    assert_allclose(inv(x.State), x.inverse().State)

def test_Exp():
    x = np.arange(1,6)

    ours = SO3["D"].Exp(x)
    theirs = expm( SO3["D"].Wedge(x) )

    assert_allclose(theirs, ours.State)
    assert_allclose(x[-2:], ours.Aug)
    with pytest.raises(Exception):
        SO3[3].Exp(x)

def test_Log():
    xi = np.array([.1, .2, .3, 5, 6])
    x = SO3["D"](xi)

    assert_allclose(xi, x.log())

def test_Wedge():
    x = np.arange(1,4)
    ours = SO3["D"].Wedge(x)
    theirs =  np.array([[0, -3,  2],
                        [3,  0, -1],
                        [-2, 1,  0]])
    
    assert_allclose(theirs, ours)

    with pytest.raises(Exception):
        SO3[3].Wedge(x)

def test_Adjoint():
    x = SO3[1](1,1,1)
    expected = np.eye(4)
    expected[0:3,0:3] = x.State
    assert_allclose(expected, x.Ad())
    assert_allclose(expected, SO3[1].Adjoint(x))