import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
from TempReg import TempReg


env = TempReg()
env.reset()
actionSpace = env.action_space
actionSize = env.action_space.n
stateSize = env.observation_space.shape[0]
print("The action size is {}, the state size is {}".format(actionSize, stateSize))
agent = Agent(state_size=stateSize, action_size=actionSize, action_space= actionSpace, seed=0)


def dqn(n_episodes=20000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    plot_score = []
    mark = []
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        #         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            plot_score.append(np.mean(scores_window))
            mark.append(i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 50.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            # torch.save(agent.actor.state_dict(), envName + 'checkpoint.pth')
            break
        if i_episode == n_episodes:
            print('\nTraining done after {:d} episodes!\t Average Score of last 100 episodes: {:.2f}'.format(i_episode,
                                                                                                             np.mean(
                                                                                                                 scores_window)))
            # torch.save(agent.actor.state_dict(), envName + 'checkpoint_20.pth')
    return plot_score, mark


score, time = dqn()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(time, score, linewidth=2)
plt.ylabel('Score', fontsize=16)
plt.xlabel('Episode #', fontsize=16)
plt.legend()
plt.show()

