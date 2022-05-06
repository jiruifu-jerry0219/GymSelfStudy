import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Solving the dynamics using the odeint function
"""
m = 1
L = 1
b = 0
g = 9.81
delta_t = 0.02
t_max = 5
theta1_0 = np.pi/2 #starting point of the pendulum
theta2_0 = 0 #initial velocity
theta_init = [theta1_0, theta2_0]
t = np.linspace(0, t_max, int(t_max/delta_t))
print(t)
# Simulate the pendulum through numerical integration
"""
Build a function for the dynamics of the pendulum dx/dt = Ax(t) + u(t)
"""
def int_pendulum_sim(y, t, L = 1, m = 1, b = 0, g = 9.81):
    theta_dot_1, theta_dot_2 = y
    dydt = [theta_dot_2, -b * theta_dot_2 - m * g * np.sin(theta_dot_1)]
    return dydt

"""
Solving the dynamics using the Euler Method
https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
"""
def euler_method(y, t, L = 1, m = 1, b = 0, g = 9.81):
    y_1 = [y[0]]
    y_2 = [y[1]]
    dt = t[1] - t[0]
    # Enumerate through time array except the end point
    for i, t_ in enumerate(t[:-1]):
        next_y1 = y_1[-1] + y_2[-1] * dt
        next_y2 = y_2[-1] - (b / (m * L ** 2) * y_2[-1] - g/L * np.sin(next_y1)) * dt
        y_1.append(next_y1)
        y_2.append(next_y2)
    dydt = np.stack([y_1, y_2]).T
    return dydt

"""
Rieman Method
"""
def rieman_method(y, t, L = 1, m = 1, b = 0, g = 9.81):
    y_1 = [y[0]]
    y_2 = [y[1]]
    for i, t_ in enumerate(t[:-1]):
        h = t[i+1] - t_
        next_1 = 


"""
Validate both methods
"""

sol1 = odeint(int_pendulum_sim, theta_init, t, args=(L, m, b, g))

theta = sol1[:, 0]
thetad = sol1[:, 1]

sol2 = euler_method(theta_init, t)
y1 = sol2[:, 0]
y2 = sol2[:, 1]

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, theta)
axs[0, 0].set_ylabel('Angles')
axs[1, 0].plot(t, thetad)
axs[1, 0].set_xlabel('time (s)')
axs[1, 0].set_ylabel('Angular Velocity')
axs[0, 1].plot(t, y1,color='r')
# axs[0, 1].set_ylabel('Angles')
axs[1, 1].plot(t, y2, color='r')
axs[1, 1].set_xlabel('time (s)')
# axs[1, 1].set_ylabel('Angular Velocity')
plt.show()






