import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-darkgrid')


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=5.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

        # Plotting definitions
        # running plot window, in seconds, or float("inf")
        self.plt_window = float("inf")

    def run(self, x0=[0, 0, 0, 0]):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time /
                              self.dt) + 1  # account for 0th
        t = np.array([0])
        y_vec = np.array([x0]).T
        u_vec = np.zeros([4, 1])

        # Start figure
        if len(x0) == 12 or len(x0) > 12:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        else:
            print("Check your state dimensions.")
            exit()
        for i in range(sim_loop_length):

            # Translate data to ca.DM
            x = ca.DM(np.size(y_vec, 0), 1).full()
            x = np.array([y_vec[:, -1]]).T

            # Get control input and obtain next state

            try:
                u = self.controller(x, t[-1])
                time.sleep(0.5)
                x_next = self.dynamics(x, u)
            except RuntimeError as e:
                print("Uh oh, your simulator crashed due to unstable dynamics.\n \
                    Retry with new controller parameters.")
                print(e)
                exit()

            # Store data
            t = np.append(t, t[-1] + self.dt)
            y_vec = np.append(y_vec, np.array(x_next), axis=1)
            u_vec = np.append(u_vec, np.array(u), axis=1)

            # Get plot window values:
            if self.plt_window != float("inf"):
                l_wnd = 0 if int(
                    i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
            else:
                l_wnd = 0

            if len(x0) == 12 or len(x0) > 12:
                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.plot(t[l_wnd:-1], y_vec[0, l_wnd:-1], label=r"$x$")
                ax1.plot(t[l_wnd:-1], y_vec[1, l_wnd:-1], label=r"$y$")
                ax1.plot(t[l_wnd:-1], y_vec[2, l_wnd:-1], label=r"$z$")
                ax1.set_ylabel("Positions")
                ax1.legend()

                ax2.plot(t[l_wnd:-1], y_vec[3, l_wnd:-1]
                         * 180 / np.pi, label=r"$\theta$")
                ax2.plot(t[l_wnd:-1], y_vec[4, l_wnd:-1]
                         * 180 / np.pi, label=r"$\phi$")
                ax2.plot(t[l_wnd:-1], y_vec[5, l_wnd:-1]
                         * 180 / np.pi, label=r"$\psi$")
                ax2.set_ylabel(r"Angles [deg]")
                ax2.legend()

                # u
                ax3.plot(t[l_wnd:-1], u_vec[0, l_wnd:-1], label='fz')
                ax3.plot(t[l_wnd:-1], u_vec[1, l_wnd:-1], label=r'$\tau_1$')
                ax3.plot(t[l_wnd:-1], u_vec[2, l_wnd:-1], label=r'$\tau_2$')
                ax3.plot(t[l_wnd:-1], u_vec[3, l_wnd:-1], label=r'$\tau_3$')
                ax3.legend()
                ax3.set_xlabel("Time [s]")
                ax3.set_ylabel("Control [u]")

                fig.suptitle(f'Quadrotor - reference tracking')
                '''

                The control will seem oscillatory but that is fine because we are calculateing
                it from the refrence frame of the drone. When the drone is upside down the gravity
                will point in its postive z-direction.
                '''

            else:
                print("Please check your state dimensions.")
                exit()

            plt.pause(0.01)

        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot(y_vec[0, l_wnd:-1],
                  y_vec[1, l_wnd:-1], y_vec[2, l_wnd:-1])
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        lim_min = np.amin([
            np.amin(y_vec[0, l_wnd:-1]),
            np.amin(y_vec[1, l_wnd:-1]),
            np.amin(y_vec[2, l_wnd:-1])])
        lim_max = np.amax([
            np.amax(y_vec[0, l_wnd:-1]),
            np.amax(y_vec[1, l_wnd:-1]),
            np.amax(y_vec[2, l_wnd:-1])])

        ax3d.set_xlim3d(lim_min, lim_max)
        ax3d.set_ylim3d(lim_min, lim_max)
        ax3d.set_zlim3d(lim_min, lim_max)
        plt.show()
        return t, y_vec, u_vec

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window
