#!/usr/bin/python3

import numpy as np

class HeadingCalculator():

    def __init__(self, radius=5):
        self.radius_to_use_desired_heading = radius
        self.x_d = 0.0
        self.y_d = 0.0
        self.psi_d = 0.0

    def update_waypoint(self, x_d, y_d, psi_d):
        self.x_d = x_d
        self.y_d = y_d
        self.psi_d = psi_d

    def update_position(self, x, y):
        distance = np.sqrt((x-self.x_d)**2 + (y-self.y_d)**2)
        if distance > self.radius_to_use_desired_heading:
            return np.arctan2(self.y_d-y, self.x_d-x)
        else:
            return self.psi_d
