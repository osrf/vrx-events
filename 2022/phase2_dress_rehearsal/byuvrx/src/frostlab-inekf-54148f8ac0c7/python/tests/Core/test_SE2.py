import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from numpy.linalg import inv
from inekf import SE2
import pytest

def test_BaseConstructor1():
    state = np.eye(4)
    sigma = np.eye(7)
    aug = np.zeros(2)

    x = SE2[2,2](state, sigma, aug)

    assert_allclose(state, x.State)
    assert_allclose(sigma, x.Cov)
    assert_allclose(aug, x.Aug)

def test_BaseConstructor2():
    state = np.eye(4)
    sigma = np.eye(6)
    aug = np.zeros(1)

    with pytest.raises(Exception):
        x = SE2["D"](state, sigma)

    x = SE2["D",1](state, sigma, aug)
    x = SE2[2,"D"](state, sigma, aug)

def test_TangentConstructor1():
    x = np.arange(6)

    state = SE2[2,1](x)

    assert_allclose(state.R().State, np.eye(2))
    assert state.State[0,2] == 1
    assert state.State[1,2] == 2
    assert state.State[0,3] == 3
    assert state.State[1,3] == 4
    assert state.Aug[0] == 5

def test_TangentConstructor2():
    x = np.arange(6)

    state = SE2["D", 1](x)
    assert_allclose(state.R().State, np.eye(2))
    assert state.State[0,2] == 1
    assert state.State[1,2] == 2
    assert state.State[0,3] == 3
    assert state.State[1,3] == 4
    assert state.Aug[0] == 5

def test_PlainConstructor():
    x = SE2(0,1,2)
    assert_allclose(x.R().State, np.eye(2))
    assert x.State[0,2] == 1
    assert x.State[1,2] == 2

def test_AddCol():
    x = SE2["D",0]()
    assert_allclose(x.State, np.eye(3))

    x.addCol(np.ones(2))

    assert_allclose(x[1], np.ones(2))

def test_AddAug():
    x = SE2[1,"D"]()

    x.addAug(2)
    assert x.Aug[-1] == 2

    with pytest.raises(Exception):
        y = SE2()
        y.addAug(2)

def test_Inverse():
    x = SE2()
    assert_allclose(inv(x.State), x.inverse().State)

def test_Exp():
    x = np.arange(6)

    ours = SE2[2,1].Exp(x)
    theirs = expm( SE2[2,1].Wedge(x) )

    assert_allclose(theirs, ours.State)
    assert x[-1] == ours.Aug[0]
    with pytest.raises(Exception):
        SE2["D",2].Exp(x)

def test_Log():
    xi = np.array([.1, 2, 3])
    x = SE2.Exp(xi)

    assert_allclose(x.log(), xi)

def test_Wedge():
    x = np.arange(1,7)
    ours = SE2[2,1].Wedge(x)
    theirs =  np.array([[0, -1, 2, 4],
                        [1, 0, 3, 5],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
    
    assert_allclose(theirs, ours)

    with pytest.raises(Exception):
        SE2["D",2].Wedge(x)

def test_Adjoint():
    x = SE2(1, 2, 3)
    ad = x.Ad()

    assert_allclose(ad[-2:,-2:], x.R().State)
    assert ad[0,0] == 1
    assert ad[1,0] == 3
    assert ad[2,0] == -2
