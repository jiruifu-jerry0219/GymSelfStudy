import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Cartpole(Env):

    def __init__(self,
                 seed=1,
                 mcart=1,
                 mpole,
                 L=1,
                 b=1,
                 max_force=1):
        """
        :param seed: Random seed
        :param mcart: mass of cart
        :param mpole: mass of pole
        :param L: length of pole
        :param b: damping coefficient
        :param max_force: maximum action
        ## Introduction of environment
        1. State space: [\theta, dtheta/dt, x, dxdt]
            a) \theta: the angle of the pole which range between (-90, 90)degree
            b) dtheta/dt: the angular velocity of the pole which range between (-18, 18) deg/s
            c) x: translational displacement of the cart which range between (-100, 100) m
            d) dx/dt: translational velocity of the cart which range between (-5, 5) m/s
        2. Action space:
            a) F: horizontal force applied to the cart which range between (-max_force, max_force) [defined by user]
        """
        self.seed = seed
        self.m_pole = mpole
        self.m_cart = mcart
        self.L = L
        self.rho = self.m_pole / self.L
        self.b = b
        self.g = 9.81
        self.dt = 0.01
        self.max_force = max_force
        self.max_theta = np.pi / 2
        self.max_thetad = np.pi/10
        self.x_max = 100
        self.v_max = 5

        self.state_bound = np.array([self.max_theta, self.max_thetad, self.x_max, self.v_max], dtype=np.float32)

        self.action_space = Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-self.state_bound, high=self.state_bound)
        self.t = None
        self.max_time = None
        self.steps_beyond_done = None

    def reset(self):
        super().reset(seed=self.seed)
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.t = 0
        self.steps_beyond_done = None
        self.max_time = 120
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        theta, thetaot, x, xdot = self.state
        force = np.clip(action, -self.max_force, self.max_force)[0]
        self.last_Force = force
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.rho * thetaot ** 2 * sintheta
        ) / (self.m_pole + self.m_cart)
        thetaacc = (self.g * sintheta)





