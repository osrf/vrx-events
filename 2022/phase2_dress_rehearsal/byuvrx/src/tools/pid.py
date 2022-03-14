class PID():
    '''
    :class PID: implements a generic PID controller

    '''
    def __init__(self, kp, ki, kd, Ts, dirty_derivative=False, beta=0.0, anti_windup_derivative_max=float('inf')):
        '''
        initialization function
        '''
        # Gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Timing
        self.Ts = Ts

        # Dirty derivative
        self.beta = beta
        self.dirty_derivative = dirty_derivative
        self.y_dot = 0.0
        self.y_d1 = 0.0

        # Memory variables
        self.integrator = 0.0
        self.error_d1 = 0.0

        # Anti windup
        self.anti_windup_derivative_max = anti_windup_derivative_max


    def update(self, x, y_r):
        '''
        Updates the controller.

        :param x: contains state information
        :param y_r: reference of the observed and tracked variable
        '''
        y = x.item(0)
        if not self.dirty_derivative:
            self.y_dot = x.item(1)
        else:
            self.y_dot = self.beta * self.y_dot \
                             + (1 - self.beta) * ((y - self.y_d1) / self.Ts)
        error = y_r - y
        # integrate error
        self.integrator = self.integrator + (self.Ts/2)*(error+self.error_d1)

        # Compute force
        u = self.kp*error + self.ki*self.integrator - self.kd*self.y_dot

        # Anti-windup
        if abs(self.y_dot) > abs(self.anti_windup_derivative_max):
            self.integrator -= (self.Ts/2)*(error+self.error_d1)

        # Update delayed error
        self.error_d1 = error
        self.y_d1 = y

        return u