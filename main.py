import numpy as np
import scipy

from model import Quadrotor
from mpc import MPC
from simulation import EmbeddedSimEnvironment

# Create quadrotor and controller objects
quad = Quadrotor(mode='nonlin')

# # Get the system discrete-time dynamics
# A, B, C = pendulum.get_discrete_system_matrices_at_eq()

# # Solve the ARE for our system to extract the terminal weight matrix P
Q = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
R = np.diag([1, 1, 1, 1])
P = 1

# tc_lb = np.array([-0.1898,-1.571])
# tc_lb =0
# tc_ub = np.array([0.1898,1.571])
# tc_ub = 0

# Instantiate controller
ctl = MPC(model=quad,
          dynamics=quad.discrete_time_dynamics,
          Q=Q, R=R, P=P,
          horizon=5,
          #   ulb=-10, uub=10,
          #   xlb=[],
          #   xub=[np.pi / 4, np.pi / 2],
          terminal_constraint=True,
          terminal_constraint_lb=0,
          terminal_constraint_ub=0)

# Part NON LINEAR DYNAMICS TESTED


def dummy_ctrl(x, u0=None):
    # return np.zeros([4, 1])
    return np.array([[0.1, -0.000, 0.001, 0.000]]).T


sim_env = EmbeddedSimEnvironment(model=quad,
                                 dynamics=quad.discrete_time_dynamics,
                                 controller=dummy_ctrl,
                                 time=5)
# ctl.set_reference([0 * 12))
t, y, u = sim_env.run([0] * 12)

# Part III - Full cart model
# pendulum = Pendulum()

# # Get the system discrete-time dynamics
# A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()

# # Solve the ARE for our system to extract the terminal weight matrix P
# Q = np.eye(4)*1
# R = np.eye(1)*1
# P = np.eye(4)*1

# # Instantiate controller
# ctl = MPC(model=pendulum,
#           dynamics=pendulum.discrete_time_dynamics,
#           horizon=7,
#           Q = Q , R = R, P = P,
#           ulb=-5, uub=5,
#           xlb=[-2, -10, -np.pi/2, -np.pi/2],
#           xub=[12, 10, np.pi/2, np.pi/2])

# # Solve without disturbance
# ctl.set_reference(x_sp=np.array([10,0,0,0]))
# sim_env_full = EmbeddedSimEnvironment(model=pendulum,
#                                 dynamics=pendulum.pendulum_linear_dynamics_with_disturbance,
#                                 controller=ctl.mpc_controller,
#                                 time = 6)
# sim_env_full.run([0,0,0,0])

# # Solve witho disturbance
# pendulum.enable_disturbance(w=0.05)
# ctl.set_reference(x_sp=np.array([10,0,0,0]))
# sim_env_full_dist = EmbeddedSimEnvironment(model=pendulum,
#                                 dynamics=pendulum.discrete_time_dynamics,
#                                 controller=ctl.mpc_controller,
#                                 time = 10)
# sim_env_full_dist.set_window(5)
# sim_env_full_dist.run([0,0,0,0])
