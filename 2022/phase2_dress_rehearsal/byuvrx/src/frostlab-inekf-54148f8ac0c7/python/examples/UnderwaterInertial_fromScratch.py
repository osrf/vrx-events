import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from inekf import SE3, SO3, InEKF, ERROR, MeasureModel, ProcessModel

np.set_printoptions(suppress=True, linewidth=300)
"""
In this one, we inherit the necessary models from the library,
and implement them in python, rather than using the pre-existing C++ implementations.

This is mostly for demonstration purposes (and to double check everything works.)

"""
class InertialProcess(ProcessModel[SE3[2,6], "Vec6"]):
    def __init__(self):
        super().__init__()
        self.g = np.array([0, 0, -9.81])
        self.Q = np.zeros((15,15))
        
    def f(self, u, dt, state):
        u_shifted = u - state.Aug
        omega = u_shifted[:3]
        a = u_shifted[3:]
        R = state.R().State
        v = state[0]
        p = state[1]

        S = np.eye(5)
        S[:3,:3] = R @ SO3.Exp(omega*dt).State
        S[:3,3]  = v + (R@a + self.g)*dt
        S[:3,4]  = p + v*dt + (R@a + self.g)*dt**2/2

        state.State = S

        return state

    def MakePhi(self, u, dt, state, error):
        A = np.zeros((15,15))

        if error == ERROR.RIGHT:
            R = state.R().State
            v_cross = SO3.Wedge( state[0] )
            p_cross = SO3.Wedge( state[1] )

            A[3:6,0:3] = SO3.Wedge(self.g)
            A[6:9,3:6] = np.eye(3)
            A[0:3,9:12] = -R
            A[3:6,9:12] = -v_cross@R
            A[6:9,9:12] = -p_cross@R
            A[3:6,12:15] = -R

            return np.eye(15) + A*dt + A@A*dt**2/2 + A@A@A*dt**3/6

        elif error == ERROR.LEFT:
            u_shifted = u - state.Aug
            w_cross = SO3.Wedge( u_shifted[0:3] )
            a_cross = SO3.Wedge( u_shifted[3:6] )

            A[0:3,0:3] = -w_cross
            A[3:6,3:6] = -w_cross
            A[6:9,6:9] = -w_cross
            A[3:6,0:3] = -a_cross
            A[6:9,3:6] = np.eye(3)

            A[0:3,9:12] = -np.eye(3)
            A[3:6,12:15] = -np.eye(3)

            return expm(A*dt)

    def setGyroNoise(self, std):
        Q = self.Q
        Q[0:3,0:3] = np.eye(3)*std**2
        self.Q = Q

    def setAccelNoise(self, std):
        Q = self.Q
        Q[3:6,3:6] = np.eye(3)*std**2
        self.Q = Q

    def setGyroBiasNoise(self, std):
        Q = self.Q
        Q[9:12,9:12] = np.eye(3)*std**2
        self.Q = Q

    def setAccelBiasNoise(self, std):
        Q = self.Q
        Q[12:15,12:15] = np.eye(3)*std**2
        self.Q = Q

class DVLSensor(MeasureModel[SE3[2,6]]):
    def __init__(self, dvlR=SO3(), dvlT=np.zeros(3)):
        super().__init__()

        self.dvlR = dvlR
        self.dvlT = SO3.Wedge(dvlT)

        self.error = ERROR.RIGHT
        self.M = np.zeros((3,3))
        H = np.zeros((3,15))
        H[0:3,3:6] = np.eye(3)
        self.H = H

    def setNoise(self, std_dvl, std_imu):
        self.M = np.eye(3)*std_dvl**2

        IMU = np.eye(3)*std_imu**2
        self.M = self.dvlR.State@self.M@self.dvlR.State.T + dvlT@IMU@dvlT.T

    def processZ(self, z, state):
        z_full = np.array([z[0], z[1], z[2], -1, 0])

        omega = z[3:6]
        z_full[:3] = self.dvlR.State@z_full[:3] + self.dvlT@omega

        return z_full

class DepthSensor(MeasureModel[SE3[2,6]]):
    def __init__(self, std):
        super().__init__()

        self.error = ERROR.LEFT

        M = np.zeros((3,3))
        M[2,2] = 1 / std**2
        self.M = M

        H = np.zeros((3,15))
        H[0:3,6:9] = np.eye(3)
        self.H = H

    def processZ(self, z, state):
        p = state[1]
        z_full = np.array([p[0], p[1], z[0], 0, 1])
        return z_full

    def calcSInverse(self, state):
        R = state.R().State
        Sig = inv(self.H_error@state.Cov@self.H_error.T)
        return (Sig - Sig@inv(R.T@self.M@R + Sig)@Sig)



# set up initial state
xi = np.array([-0.0411, -0.1232, -1.5660,   # rotation
                0, 0, -4.66134,             # velocity
                -0.077, -0.02, -2.2082,     # position
                0,0,0,0,0,0])               # bias
s = np.array([0.274156, 0.274156, 0.274156, 
                1.0, 1.0, 1.0, 
                0.01, 0.01, 0.01, 
                0.000025, 0.000025, 0.000025, 
                0.0025, 0.0025, 0.0025])
x0 = SE3[2,6](xi, np.diag(s))

# set up DVL
dvlR = SO3(np.array([[0.000, -0.995, 0.105],
                [0.999, 0.005, 0.052],
                [-0.052, 0.104, 0.993]]))
dvlT = np.array([-0.17137, 0.00922, -0.33989])
dvl = DVLSensor(dvlR, dvlT)
dvl.setNoise(.0101*2.6, .005*(3.14/180)*np.sqrt(200.0))

# set up depth sensor
depth = DepthSensor(51.0 * (1.0/100) * (1.0/2))

# setup iekf
iekf = InEKF[InertialProcess](x0, ERROR.RIGHT)
iekf.addMeasureModel('DVL', dvl)
iekf.addMeasureModel('Depth', depth)

# set process model noise
iekf.pModel.setGyroNoise( .005 *  (3.14/180)  * np.sqrt(200.0) )
iekf.pModel.setAccelNoise( 20.0 * (10**-6/9.81) * np.sqrt(200.0) )
iekf.pModel.setGyroBiasNoise(0.001)
iekf.pModel.setAccelBiasNoise(0.001)

# iterate through all of data
num = 3000
f = open('../../data/underwater_data.txt')

# create place to hold everything
actual = []
mine = []

def parse_line(l):
    l = l.split()
    name = l[0]
    dt = float(l[1])
    data = np.array([float(s) for s in l[2:]])

    return name, dt, data

for i in range(num*4):
    l = f.readline()
    name, dt, data = parse_line(l)

    if name == "IMU":
        # print("Recieved IMU")
        imu_data = data
        iekf.Predict(data, dt)
    if name == "DVL":
        # print("Recieved DVL")
        data = np.append(data, imu_data[0:3])
        state = iekf.Update(data, "DVL")
    if name == "DEPTH":
        # print("Recieved Depth")
        state = iekf.Update(data, "Depth")
        mine.append(state.State)
    
    if name == "R":
        # print("Recieved Actual")
        state = np.eye(5)
        state[0:3,0:3] = data.reshape((3,3))
        name, dt, data = parse_line(f.readline())
        state[0:3,3] = data
        name, dt, data = parse_line(f.readline())
        state[0:3,4] = data
        actual.append(state)

actual = np.array(actual)
mine = np.array(mine)
t = np.arange(num)*dt

fig, ax = plt.subplots(3,3, figsize=(12,8))
ax[0,1].set_title("Position")
ax[1,1].set_title("Velocity")
ax[2,0].set_title("Pitch")
ax[2,1].set_title("Roll")
ax[2,2].set_title("Yaw")

ax[0,0].plot(t, actual[:,0,4], label="Actual")
ax[0,0].plot(t,   mine[:,0,4], label="InEKF")
ax[0,1].plot(t, actual[:,1,4], label="Actual")
ax[0,1].plot(t,   mine[:,1,4], label="InEKF")
ax[0,2].plot(t, actual[:,2,4], label="Actual")
ax[0,2].plot(t,   mine[:,2,4], label="InEKF")

actual_inv = np.linalg.inv(actual)
mine_inv = np.linalg.inv(mine)

ax[1,0].plot(t, -actual_inv[:,0,3], label="Actual")
ax[1,0].plot(t, -  mine_inv[:,0,3], label="InEKF")
ax[1,1].plot(t, -actual_inv[:,1,3], label="Actual")
ax[1,1].plot(t, -  mine_inv[:,1,3], label="InEKF")
ax[1,2].plot(t, -actual_inv[:,2,3], label="Actual")
ax[1,2].plot(t, -  mine_inv[:,2,3], label="InEKF")

mine_p   = -np.arcsin(mine[:,2,0])
actual_p = -np.arcsin(actual[:,2,0])
mine_r   = np.arctan2(mine[:,2,1], mine[:,2,2])
actual_r = np.arctan2(actual[:,2,1], actual[:,2,2])
mine_y   = np.arctan2(mine[:,1,0], mine[:,0,0])
actual_y = np.arctan2(actual[:,1,0], actual[:,0,0])

ax[2,0].plot(t, actual_p, label="Actual")
ax[2,0].plot(t,   mine_p, label="InEKF")
ax[2,1].plot(t, actual_r, label="Actual")
ax[2,1].plot(t,   mine_r, label="InEKF")
ax[2,2].plot(t, actual_y, label="Actual")
ax[2,2].plot(t,   mine_y, label="InEKF")

plt.tight_layout()
plt.show()