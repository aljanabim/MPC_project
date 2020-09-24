import casadi as ca
import numpy as np
from filterpy.kalman import KalmanFilter


class Quadrotor(object):
    def __init__(self, h=0.1, m=None, M=None):
        """
        Quadrotor model class.
        Describes the movement of
        """
        self.g = np.array([[0, 0, -9.82]]).T

        self.dt = h

        # QUADROTOR MASS
        # set mass
        if m is None:
            self.m = 1.4
        else:
            self.m = m
        # set inertia matrix
        if M is None:
            self.M = np.diag([0.001, 0.001, 0.005])
        else:
            self.M = M

        self.set_integrators()
        self.set_discrete_time_system()
        # self.set_augmented_discrete_system()

    def model_lin(self, x, u):
        """
        Linearized dynamics of the quadrotor
        """
        pass

    def model_nonlin(self, x, u):
        """
        Nonlinear dynamics of the quadrotor
        """
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]
        x12 = x[11]

        omega = x[9:12]
        alpha = x[6:9]
        fz = u[0]
        tau = u[1:]

        p_dot = [x4, x5, x6]
        v_dot = self.R(alpha)@self.f(fz)  # / self.m + self.g
        alpha_dot = [0, 0, 0]  # self.T(alpha) @ omega
        # np.pinv(self.M)@(tau - np.cross(omega, self.M@omega))
        omega_dot = [0, 0, 0]

        dxdt = [p_dot[0],
                p_dot[1],
                p_dot[2],
                v_dot[0],
                v_dot[1],
                v_dot[2],
                alpha_dot[0],
                alpha_dot[1],
                alpha_dot[2],
                omega_dot[0],
                omega_dot[1],
                omega_dot[2]]

        return ca.vertcat(*dxdt)

    def set_integrators(self):
        # SET CASADI VARIABLES
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)

        # INTEGRATION METHOD SETTINGS
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100,
                   "tf": self.dt}

        # CREATE INTEGRATOR FOR THE LINEAR DYNAMICS
        # dae = {'x': x, 'ode': self.model_lin(x, u), 'p': ca.vertcat(u)}
        # self.integrator_lin = ca.integrator(
        #     'integrator', 'cvodes', dae, options)

        # CREATE INTEGRATOR FOR THE NONLINEAR DYNAMICS
        dae = {'x': x, 'ode': self.model_nonlin(x, u), 'p': ca.vertcat(u)}
        self.integrator_nonlin = ca.integrator(
            'integrator', 'cvodes', dae, options)

        # TODO the integrator for the augmented system with an integrator

    def set_discrete_time_system(self):
        pass

    def R(self, alpha=[0, 0, 0]):
        print(alpha)
        theta = alpha[0]
        phi = alpha[1]
        psi = alpha[2]

        R1 = ca.DM([[1, 0, 0],
                    [0, ca.cos(theta), -ca.sin(theta)],
                    [0, ca.sin(theta), ca.cos(theta)]])
        print(R1)
        exit()
        return R1
        # R2 = np.array([[np.cos(phi), 0, np.sin(phi)],
        #                [0, 1, 0],
        #                [-np.sin(phi), 0, np.cos(phi)]]).astype(np.float32)

        # R3 = np.array([[np.cos(psi), -np.sin(psi), 0],
        #                [np.sin(psi), np.cos(psi), 0],
        #                [0, 0, 1]]).astype(np.float32)
        # return R1@R2@R3

    @staticmethod
    def T(alpha=[0, 0, 0]):
        theta = alpha[0]
        phi = alpha[1]
        psi = alpha[2]
        return np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                         [0, np.cos(phi), -np.sin(phi)],
                         [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])

    @staticmethod
    def f(fz=0):
        f = ca.MX(3, 1)
        f[2] = fz
        return f


Quadrotor()
