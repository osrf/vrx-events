import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, logm
from numpy.linalg import inv
from inekf import SE3, SO3
import pytest

def test_BaseConstructor1():
    state = np.eye(5)
    sigma = np.eye(11)
    aug = np.zeros(2)

    x = SE3[2,2](state, sigma, aug)

    assert_allclose(state, x.State)
    assert_allclose(sigma, x.Cov)
    assert_allclose(aug, x.Aug)

def test_BaseConstructor2():
    state = np.eye(5)
    sigma = np.eye(10)
    aug = np.zeros(1)

    with pytest.raises(Exception):
        x = SE3["D"](state, sigma)

    x = SE3["D",1](state, sigma, aug)
    x = SE3[2,"D"](state, sigma, aug)

def test_TangentConstructor1():
    x = np.arange(10)

    state = SE3[2,1](x)

    assert_allclose(state.R().State, SO3.Exp(x[:3]).State)
    assert_allclose(state[0], x[3:6])
    assert_allclose(state[1], x[6:9])
    assert state.Aug[0] == x[-1]

def test_TangentConstructor2():
    x = np.arange(10)

    state = SE3["D", 1](x)
    assert_allclose(state.R().State, SO3.Exp(x[:3]).State)
    assert_allclose(state[0], x[3:6])
    assert_allclose(state[1], x[6:9])
    assert state.Aug[0] == x[-1]

    state = SE3[2,"D"](x)
    assert_allclose(state.R().State, SO3.Exp(x[:3]).State)
    assert_allclose(state[0], x[3:6])
    assert_allclose(state[1], x[6:9])
    assert state.Aug[0] == x[-1]

def test_PlainConstructor():
    x = SE3(0,0,0,4,5,6)
    assert_allclose(x.R().State, np.eye(3))
    assert_allclose(x[0], [4,5,6])

def test_AddCol():
    x = SE3["D",0]()
    assert_allclose(x.State, np.eye(4))

    x.addCol(np.ones(3))

    assert_allclose(x[1], np.ones(3))

def test_AddAug():
    x = SE3[1,"D"]()

    x.addAug(2)
    assert x.Aug[-1] == 2

    with pytest.raises(Exception):
        y = SE3()
        y.addAug(2)

def test_Inverse():
    x = SE3()
    assert_allclose(inv(x.State), x.inverse().State)

def test_Exp():
    x = np.arange(10)

    ours = SE3[2,1].Exp(x)
    theirs = expm( SE3[2,1].Wedge(x) )

    assert_allclose(theirs, ours.State)
    assert x[-1] == ours.Aug[0]
    with pytest.raises(Exception):
        SE3["D",2].Exp(x)

def test_Log():
    xi = np.array([.1, .2, .3, 4, 5, 6])
    x = SE3.Exp(xi)

    assert_allclose(x.log(), xi)

def test_Wedge():
    x = np.arange(1,11)
    ours = SE3[2,1].Wedge(x)
    theirs =  np.array([[0,  -3,  2,  4,  7],
                        [3,   0, -1,  5,  8],
                        [-2,  1,  0,  6,  9],
                        [0,   0,  0,  0,  0],
                        [0,   0,  0,  0,  0]])
    
    assert_allclose(theirs, ours)

    with pytest.raises(Exception):
        SE3["D",2].Wedge(x)

def test_Adjoint():
    x = np.arange(1,11)
    x = SE3[2,1](x)
    ad = x.Ad()

    for i in range(3):
        assert_allclose(ad[3*i:3*i+3, 3*i:3*i+3], x.R().State)
    
    for i in range(2):
        top = SO3.Wedge(x[i])@x.R().State
        assert_allclose(ad[3+3*i:6+3*i,0:3], top)

    assert ad[9,9] == 1
