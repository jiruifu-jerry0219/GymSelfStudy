import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Cartpole(Env):


    def __init__(self, seed, low, high):
        """

        :param seed: Define the random seed
        :param low: The minimum value of the action space
        :param high: The maximum value of the action space
        """
        self.action_space = Box(low=-2, high=2)
        self.observation_space = Box(low=np.array([-np.pi, 0, 0, 0]), high=np.array([np.pi, np.pi / 10, 100, 10]))
        """
        State space: angle, angular velocity, translational displacement, translational velocity
        """
        self.state =

