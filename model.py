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
        self.x_d = None
        # QUADROTOR MASS
        # set mass
        if m is None:
            self.m = 1.4
        else:
            self.m = m
        # set inertia matrix
        if M is None:
            self.M = ca.diag([0.001, 0.001, 0.005])
        else:
            self.M = M
        self.x_eq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.u_eq = [0, 0, 0, 0]

        self.set_integrators()
        self.set_discrete_time_system()
        self.set_discrete_time_aug_system()

    def model_lin(self, x, u):
        """
        Linearized dynamics of the quadrotor
        """
        M_x = self.M[0, 0]
        M_y = self.M[1, 1]
        M_z = self.M[2, 2]

        w_x = self.x_eq[9]
        w_y = self.x_eq[10]
        w_z = self.x_eq[11]

        g = -self.g[2]

        Ac = ca.MX.zeros(12, 12)
        # First Row
        Ac[0, 3] = 1
        # Second row
        Ac[1, 4] = 1
        # third row
        Ac[2, 5] = 1
        # forth row
        Ac[3, 7] = g
        # fifth row
        Ac[4, 6] = -g
        # 6:th row = Empty
        # 7:th row
        Ac[6, 7] = w_z
        Ac[6, 9] = 1
        # 8:th row
        Ac[7, 6] = -w_z
        Ac[7, 10] = 1
        # 9:th row
        Ac[8, 6] = w_y
        Ac[8, 11] = 1
        # 10:th row
        Ac[9, 10] = (-w_z * M_z + w_z * M_y) / M_x
        Ac[9, 11] = (-w_y * M_z + w_y * M_y) / M_x
        # 11:th row
        Ac[10, 9] = (w_z * M_z - w_z * M_x) / M_y
        Ac[10, 11] = (w_x * M_z - w_x * M_x) / M_y
        # 12:th row
        Ac[11, 9] = (-w_y * M_y + w_y * M_x) / M_z
        Ac[11, 10] = (-w_x * M_y + w_x * M_x) / M_z

        Bc = ca.MX.zeros(12, 4)
        Bc[5, 0] = 1 / self.m
        Bc[9, 1] = 1 / M_x
        Bc[10, 2] = 1 / M_y
        Bc[11, 3] = 1 / M_z

        return Ac @ x + Bc @ u

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
        v_dot = self.R(alpha)@self.f(fz) / self.m + self.g
        alpha_dot = self.T(alpha) @ omega
        omega_dot = ca.pinv(self.M) @(
            tau - ca.cross(omega, self.M@omega))

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
        x = ca.MX.sym('x', len(self.x_eq))
        u = ca.MX.sym('u', len(self.u_eq))

        # INTEGRATION METHOD SETTINGS
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100,
                   "tf": self.dt}

        # CREATE INTEGRATOR FOR THE LINEAR DYNAMICS
        dae = {'x': x, 'ode': self.model_lin(x, u), 'p': ca.vertcat(u)}
        self.integrator_lin = ca.integrator(
            'integrator', 'cvodes', dae, options)

        # CREATE INTEGRATOR FOR THE NONLINEAR DYNAMICS
        dae = {'x': x, 'ode': self.model_nonlin(x, u), 'p': ca.vertcat(u)}
        self.integrator_nonlin = ca.integrator(
            'integrator', 'cvodes', dae, options)

        # TODO the integrator for the augmented system with an integrator

    def set_discrete_time_system(self):
        # Check for integrator definition
        if self.integrator_lin is None:
            print("Integrator not defined. Set integrators first.")
            exit()

        # Set CasADi variables
        x = ca.MX.sym('x', len(self.x_eq))
        u = ca.MX.sym('u', len(self.u_eq))

        # Jacobian of exact discretization
        self.Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
            self.integrator_lin(x0=x, p=u)['xf'], x)])
        self.Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
            self.integrator_lin(x0=x, p=u)['xf'], u)])

    def set_discrete_time_aug_system(self):
        print(len(self.x_eq), len(self.u_eq))
        Ad_eq = self.Ad(self.x_eq, self.u_eq)
        Bd_eq = self.Bd(self.x_eq, self.u_eq)

        C = ca.MX.eye(12)

        self.Ai = ca.MX.zeros(24, 24)
        self.Ai[0:12, 0:12] = Ad_eq
        self.Ai[12:, 0:12] = -self.dt * C
        self.Ai[12:, 12:] = ca.MX.eye(12)

        self.Bi = ca.MX.zeros(24, 4)
        self.Bi[:12, :] = Bd_eq

        self.Br = ca.MX.zeros(24, 12)
        self.Br[12:, :] = self.dt * ca.MX.eye(12)

    def discrete_time_aug_dynamics(self, x0, u):
        # print(self.Ai.shape, x0.shape, self.Bi.shape,
        #       u.shape, self.Br.shape, self.x_d.shape)
        if self.x_d is None:
            print("What are you doing with your life? SET A REFERENCE!")
            quit()
        return self.Ai @ x0 + self.Bi @ u + self.Br @ self.x_d

    def discrete_time_dynamics(self, x0, u):
        """
        Performs a discrete time iteration step.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: next discrete time state
        :rtype: 4x1, ca.DM
        """
        # if self.mode == 'nonlin':
        # alpha = x0[6:9]
        # if type(alpha[0]) == type(np.array([])):
        #     gravity_compensator = (self.R_num(alpha)@(self.g * self.m))[-1]
        #     u[0] = u[0] - gravity_compensator
        # print(u[0])
        # print()
        # print((ca.pinv(self.R(alpha))@self.g)[0].shape, u[0].shape)
        # print(ca.DM(self.g[-1] * self.m) + u[0])
        # u[0] = u[0] - ca.DM(self.g[-1] * self.m)
        # out = self.integrator_nonlin(x0=x0, p=u)
        # return out["xf"]

        # elif self.mode == 'lin':

        # else:
        #     print("MAKE UP YOUR MIND!!!! I AM OUT")
        return self.Ad(self.x_eq, self.u_eq) @ x0 + \
            self.Bd(self.x_eq, self.u_eq) @ u

    def discrete_time_nl_dynamics(self, x0, u):
        alpha = x0[6:9]
        if type(alpha[0]) == type(np.array([])):
            gravity_compensator = (self.R_num(alpha)@(self.g * self.m))[-1]
            u[0] = u[0] - gravity_compensator
        # u[0] = u[0] - ca.DM(self.g[-1] * self.m)
        out = self.integrator_nonlin(x0=x0, p=u)
        return out["xf"]

    def set_reference(self, ref):
        """
        Simple method to set the new system reference.

        :param ref: desired reference [m]
        :type ref: float or casadi.DM 1x1
        """
        self.x_d = ref

    def set_equilibrium_point(self, x_eq, u_eq):
        """
        Set a different equilibrium poin for the system.

        :param x_eq: state equilibrium
        :type x_eq: list with 4 floats, [a,b,c,d]
        :param u_eq: control input for equilibrium point
        :type u_eq: float
        """

        self.x_eq = x_eq
        self.u_eq = u_eq

        self.set_integrators()
        self.set_discrete_time_system()

    @staticmethod
    def R(alpha=[0, 0, 0]):
        theta = alpha[0]
        phi = alpha[1]
        psi = alpha[2]

        R1 = ca.MX(3, 3)
        R1[0, 0] = 1
        R1[1, 1] = ca.cos(theta)
        R1[1, 2] = -ca.sin(theta)
        R1[2, 1] = ca.sin(theta)
        R1[2, 2] = ca.cos(theta)

        R2 = ca.MX(3, 3)
        R2[1, 1] = 1
        R2[0, 0] = ca.cos(phi)
        R2[0, 2] = ca.sin(phi)
        R2[2, 0] = -ca.sin(phi)
        R2[2, 2] = ca.cos(phi)

        R3 = ca.MX(3, 3)
        R3[2, 2] = 1
        R3[0, 0] = ca.cos(psi)
        R3[0, 1] = -ca.sin(psi)
        R3[1, 0] = ca.sin(psi)
        R3[1, 1] = ca.cos(psi)

        return R3@R2@R1

    @staticmethod
    def R_num(alpha=[0, 0, 0]):
        theta = alpha[0]
        phi = alpha[1]
        psi = alpha[2]

        R1 = ca.DM(3, 3)
        R1[0, 0] = 1
        R1[1, 1] = ca.cos(theta)
        R1[1, 2] = -ca.sin(theta)
        R1[2, 1] = ca.sin(theta)
        R1[2, 2] = ca.cos(theta)

        R2 = ca.DM(3, 3)
        R2[1, 1] = 1
        R2[0, 0] = ca.cos(phi)
        R2[0, 2] = ca.sin(phi)
        R2[2, 0] = -ca.sin(phi)
        R2[2, 2] = ca.cos(phi)

        R3 = ca.DM(3, 3)
        R3[2, 2] = 1
        R3[0, 0] = ca.cos(psi)
        R3[0, 1] = -ca.sin(psi)
        R3[1, 0] = ca.sin(psi)
        R3[1, 1] = ca.cos(psi)

        return R3@R2@R1

        # R1 = ca.DM([[1, 0, 0],
        #             [0, ca.cos(theta), -ca.sin(theta)],
        #             [0, ca.sin(theta), ca.cos(theta)]])
        # print(R1)
        # exit()
        # return R1
        # R2 = np.array([[ca.cos(phi), 0, ca.sin(phi)],
        #                [0, 1, 0],
        #                [-ca.sin(phi), 0, ca.cos(phi)]]).astype(np.float32)

        # R3 = np.array([[ca.cos(psi), -ca.sin(psi), 0],
        #                [ca.sin(psi), ca.cos(psi), 0],
        #                [0, 0, 1]]).astype(np.float32)
        # return R1@R2@R3

    @staticmethod
    def T(alpha=[0, 0, 0]):
        theta = alpha[0]
        phi = alpha[1]
        psi = alpha[2]

        T = ca.MX(3, 3)
        T[0, 0] = 1
        T[1, 0] = 0
        T[2, 0] = 0
        T[0, 1] = ca.sin(theta) * ca.tan(phi)
        T[1, 1] = ca.cos(theta)
        T[2, 1] = ca.sin(theta) / ca.cos(phi)
        T[0, 2] = ca.cos(theta) * ca.tan(phi)
        T[1, 2] = -ca.sin(theta)
        T[2, 2] = ca.cos(theta) / ca.cos(phi)

        return T

    @staticmethod
    def f(fz=0):
        f = ca.MX(3, 1)
        f[2] = fz
        return f
