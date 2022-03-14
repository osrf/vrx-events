import numpy as np

class State():

    def __init__(self, x, y, psi):
        self._x = x
        self._y = y
        self._psi = psi

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def psi(self):
        return self._psi

    @property
    def position(self):
        return np.array([[self._x, self._y, 0]]).T

    def __str__(self):
        return f'position: ({self._x}, {self._y})\n' + \
               f'heading: {self._psi}'