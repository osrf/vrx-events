import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.markers as mmarkers
from scipy.stats.distributions import chi2
from time import time

np.set_printoptions(suppress=True, linewidth=300)

from inekf import OdometryProcessDynamic, LandmarkSensor, GPSSensor
from inekf import SE2, InEKF, ERROR

def data_association(state, zs, laser_mm):
    n_lm = state.State.shape[0] - 2 - 1
    n_mm = len(zs)

    # if we don't have anything, say they're all new!
    if n_lm == 0:
        return [-1]*n_mm

    alpha = chi2.ppf(0.95, 2)
    beta  = chi2.ppf(0.99, 2)
    A = np.ones((n_mm, n_mm))*alpha
    M = np.zeros((n_mm, n_lm))
    for i in range(0,n_lm):
        laser_mm.sawLandmark(i, state)
        for j, z in enumerate(zs):
            M[j,i] = laser_mm.calcMahDist(z, state)

    M_new = np.hstack((M,A))
    pairs = solve_cost_matrix_heuristic(M_new)
    pairs.sort()

    assoc = [x[1] for x in pairs]

    for i in range(len(assoc)):
        # Check if it wants a new landmark
        if assoc[i] >= n_lm:
            assoc[i] = -1
            # Also check if there was any other landmarks semi-close to it
            # if so, don't make a new one
            for j in range(M.shape[1]):
                if M[i, j] < beta:
                    assoc[i] = -2
                    break

    return assoc

def solve_cost_matrix_heuristic(M):
    n_msmts = M.shape[0]
    result = []

    ordering = np.argsort(M.min(axis=1))

    for msmt in ordering:
        match = np.argmin(M[msmt,:])
        M[:, match] = 1e8
        result.append((msmt, match))

    return result

def addLandmark(z, state):
    x, y = state[0]
    phi = np.arctan2(state.State[1,0],state.State[0,0])
    r, b = z

    xl = x + r*np.cos(b+phi)
    yl = y + r*np.sin(b+phi)

    state.addCol([xl, yl], np.eye(2)*1000)

def x(l):
    return np.array([i[0][0] for i in l])

def y(l):
    return np.array([i[0][1] for i in l])

def makeOdometry(u, dt):
    a = 3.78
    b = 0.50
    L = 2.83
    H = 0.76

    Ve, alpha = u
    Vc = Ve / (1 - np.tan(alpha)*H/L)

    motion = np.zeros(3)
    motion[0] = Vc/L*np.tan(alpha)
    motion[1] = Vc - Vc/L*np.tan(alpha)*b
    motion[2] = Vc/L*np.tan(alpha)*a

    return SE2(dt*motion)

#### SETUP INITIAL STATE
x0 = [45*np.pi/180, 0, 0]
sig = np.diag([.1,.1,.1])
x0 = SE2["D"](x0, sig)

#### SETUP InEKF
gps = GPSSensor(3)
laser = LandmarkSensor(0.5, 0.5*np.pi/180)

iekf = InEKF[OdometryProcessDynamic](x0, ERROR.RIGHT)
iekf.addMeasureModel("GPS", gps)
iekf.addMeasureModel("Laser", laser)
Q = np.diag([0.5*np.pi/180, 0.05, 0.05])
iekf.pModel.setQ(Q)

#### LOAD IN DATA
def read_data(filename):
    with open(filename, 'r') as f:
        data = [[float(x) for x in line.strip().split('\t')] for line in f.readlines()]
    return data

odo_data = np.array(read_data("../../data/victoria_park_ascii/DRS.txt"))
gps_data = np.array(read_data("../../data/victoria_park_ascii/GPS.txt"))
laser_data = read_data("../../data/victoria_park_ascii/LASER_TREE.txt")

events =      [('gps',   x[0], x[1:]) for x in gps_data]
events.extend([('odo',   x[0], x[1:]) for x in odo_data])
events.extend([('laser', x[0], np.array(x[1:]).reshape(-1,2)) for x in laser_data])
events = sorted(events, key=lambda x: x[1])

#### GET PLOT READY
fig, ax = plt.subplots(figsize=(8,6))

traj, = ax.plot([])
veh = ax.scatter([],[], label="Vehicle", marker=">")
gps_pts = ax.scatter([], [], label="GPS", c='g', s=3)
lm_pts  = ax.scatter([], [], label="Landmarks", c='r', s=3)
plt.legend()
plt.show(block=False)

#### ITERATE
states = [x0]
gps_data = []
last_t = 0
last_plot = 0
for i, (e, t, data) in tqdm(enumerate(events), total=len(events)):
    # If it's Odometry
    e, t, data = events[i]
    if e == 'odo':
        dt = t - last_t
        last_t = t

        u = makeOdometry(data, dt)
        s = iekf.Predict(u, dt)
        states.append(s)

    # GPS Measurement
    if e == 'gps':
        s = iekf.Update(data, "GPS")
        states[-1] = s
        gps_data.append(data)

    # Laser Measurement
    if e == 'laser':
        # identify landmarks
        assoc = data_association(iekf.state, data, laser)
        # iterate through them (note data here is still r/b)
        for idx, data in zip(assoc, data):
            if idx == -1:
                addLandmark(data, iekf.state)
                laser.sawLandmark(iekf.state.State.shape[0]-2-1-1, iekf.state)
                iekf.Update(data, "Laser")
            elif idx == -2:
                continue
            else:
                laser.sawLandmark(idx, iekf.state)
                iekf.Update(data, "Laser")

    if time() - last_plot > 1 or i == len(events)-1:
        # plot car
        phi = np.arctan2(states[-1].State[1,0],states[-1].State[0,0])
        m = mmarkers.MarkerStyle(">")
        m._transform = m.get_transform().rotate_deg(phi*180/np.pi)
        veh.set_offsets(states[-1][0])
        veh.set_paths((m.get_path().transformed(m.get_transform()),))
        
        # plot trajectory
        xn, yn = x(states), y(states)
        traj.set_data(xn, yn)

        # plot gps data
        if len(gps_data) != 0:
            curr = gps_pts.get_offsets()
            curr = np.vstack((curr, gps_data))
            gps_pts.set_offsets(curr)
            gps_data = []

        # plot landmark data
        n_lm = iekf.state.State.shape[0] - 2 - 1
        if n_lm != 0:
            lm = np.array([iekf.state[i] for i in range(1,n_lm+1)])
            lm_pts.set_offsets(lm)
        else:
            lm = np.zeros((1,2))

        # set limits
        min_x = min(min(xn), min(lm[:,0]))
        max_x = max(max(xn), max(lm[:,0]))
        min_y = min(min(yn), min(lm[:,1]))
        max_y = max(max(yn), max(lm[:,1]))

        ax.set_xlim(min_x*1.1-1, max_x*1.1+1)
        ax.set_ylim(min_y*1.1-1, max_y*1.1+1)
        plt.draw()
        plt.pause(.001)
        last_plot = time()

plt.show()