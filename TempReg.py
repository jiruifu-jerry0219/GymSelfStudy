import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

"""
Create the environment class and pass the Env to the Custom Environment there it can inherit the methods and properties 
of the Gym class
Full tutorial at: https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
"""
class TempReg(Env):

    def __init__(self):
        """
        Initialize the actions, observations, and episode length
        """
        self.action_space = Discrete(3) # Discrete spaces take in a fixed range of non-negative values.
        # The action space include three fixed operations
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Box spaces are much more flexible and allow us to pass through multiple values between specific range
        # In addition, it can hold images, audios, and data frames
        self.state = np.array([38 + random.randint(-3, 3)])
        # Episode# length
        self.shower_length = 60

    def step(self, action):
        self.state += action - 1
        self.shower_length -= 1

        # Calculate the reward
        if self.state[0] >= 37 and self.state[0]<= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0 or self.state[0] >= 41 or self.state[0]<= 36:
            done = True
        else:
            done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        pass
    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)])
        self.shower_length = 60
        return self.state

if __name__ =="__main__":
    env = TempReg()
    episode = 20
    actionSize = env.action_space.n
    stateSize = env.observation_space.shape
    print("The action size is {}, the state size is {}".format(actionSize, stateSize[0]))
    for episode in range(1, episode +1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode: {}, Score: {}'. format(episode, score))