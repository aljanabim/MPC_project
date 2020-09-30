import numpy as np
import scipy
import casadi as ca
from model import Quadrotor
from mpc import MPC
from simulation import EmbeddedSimEnvironment

# Create quadrotor and controller objects
quad = Quadrotor()

# ASTRID IF YOU CHANGE THE REFERNCE DO IT HERE
#                             x    y    z
quad.set_reference(np.array([0.5, 0.3, 0.5]))

# # Solve the ARE for our system to extract the terminal weight matrix P
Q = np.diag([1] * 15)
Q[2, 2] = 1000
Q[12, 12] = 20
Q[13, 13] = 20
Q[14, 14] = 500
R = np.diag([1, 1, 1, 1])
P = 1

# tc_lb = np.array([-0.1898,-1.571])
# tc_lb =0
# tc_ub = np.array([0.1898,1.571])
# tc_ub = 0

# Instantiate controller
m = quad.m
f_max = np.abs(4 * m * quad.g[-1])
tau_max = np.abs(17.5 * m * quad.g[-1])
tau_z_max = 0.01
ctl = MPC(model=quad,
          aug=True,
          dynamics=quad.discrete_time_aug_dynamics,
          Q=Q, R=R, P=P,
          horizon=5,
          disturb_thrust=0.5,  # ASTRID PLAY WITH THIS (THE THRUST DISTURBANCE)
          ulb=[-f_max, -tau_max, -tau_max, -tau_z_max],
          uub=[f_max, tau_max, tau_max, tau_z_max],
          xlb=[
              -np.inf,  # x
              -np.inf,  # y
              0,  # z
              -np.inf,  # vx
              -np.inf,  # vy
              -np.inf,  # vz
              -np.inf,  # theta
              -np.inf,  # phi
              -np.inf,  # psi
              -np.inf,  # theta_dot
              -np.inf,  # phi_dot
              -np.inf,  # psi_dot
              -np.inf,
              -np.inf,
              -np.inf,

          ],  # psi_dot
          xub=[np.inf,  # x
               np.inf,  # y
               np.inf,  # z
               np.inf,  # vx
               np.inf,  # vy
               np.inf,  # vz
               np.inf,  # theta
               np.inf,  # phi
               np.inf,  # psi
               np.inf,  # theta_dot
               np.inf,  # phi_dot
               np.inf,  # psi_dot
               np.inf,
               np.inf,
               np.inf,
               ],
          # xub=[np.pi / 4, np.pi / 2],
          terminal_constraint=True,
          terminal_constraint_lb=0,
          terminal_constraint_ub=0)
# Part NON LINEAR DYNAMICS TESTED


def dummy_ctrl(x, u0=None):
    # return np.zeros([4, 1])
    return np.array([[0.0, -0.000, 0.01, 0.000]]).T


quad.m = 2  # ASTRID PLAY WITH THIS (THE MASS)
quad.M = ca.DM([[0.001, 0.004, 0],
                [0.004, 0.001, 0],
                [0, 0, 0.005]])  # ASTRID PLAY WITH THIS TOO (THE INERTIA MATRIX)
sim_env = EmbeddedSimEnvironment(model=quad,
                                 dynamics=quad.discrete_time_nl_dynamics,
                                 controller=ctl.mpc_controller,
                                 time=10)

ctl.set_reference([
    quad.x_d[0],  # x
    quad.x_d[1],  # y
    quad.x_d[2],  # z
    0,  # vx
    0,  # vy
    0,  # vz
    0,  # theta
    0,  # phi
    0,  # psi
    0,  # theta_dot
    0,  # phi_dot
    0,  # psi_dot
    0,  # rx
    0,  # ry
    0   # rz
])
t, y, u = sim_env.run([0] * 15)
