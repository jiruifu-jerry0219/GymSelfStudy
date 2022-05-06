import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Cartpole(Env):

    def __init__(self,
                 seed=1,
                 m=1,
                 L=1, b=1,
                 max_force=1):
        """
        :param seed: Random seed
        :param m: mass of cart pole
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
        self.m = m
        self.L = L
        self.b = b
        self.g = 9.81
        self.max_force = max_force
        self.state_bound = np.array([np.pi / 2, np.pi / 10, 100, 5], dtype=np.float32)

        self.action_space = Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-self.state_bound, high=self.state_bound)
        self.t = 0
        self.dt = 0.02
        self.max_time = 12

    def reset(self):
        super().reset(seed=self.seed)
        self.init_theta =


