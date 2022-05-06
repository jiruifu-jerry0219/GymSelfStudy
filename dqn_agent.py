import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5) #Define the size of replay buffer
BATCH_SIZE = 64  # Size of Minibatch
GAMMA = 0.99 # Discount factor
TAU = 1e-3 # For soft update of target parameter
LR = 5e-4 # Learning Rate
UPDATE_EVERY = 4 # Frequency to update target network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, action_space, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.seed = seed

        self.actor = QNetwork(self.state_size, self.action_size, self.seed).to(device)
        self.target = QNetwork(self.state_size, self.action_size, self.seed).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=LR)

        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        :param state: the state of environment at t
        :param action: action a(t)
        :param reward: reward signal r(t+1)
        :param next_state: the state of environment at t+1
        :param done: True if the episode is terminated
        :return:
        """
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps = 0.):
        """
        :param state: the state of environment at t
        :param eps: epison for epsilon-greedy
        :return: action a(t)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def learn(self, experiences, gamma):
        """
        :param experiences: a tuple (states, actions, rewards, next_state, done)
        :param gamma: discount factor
        :return:
        """
        states, actions, rewards, next_state, done = experiences
        # print('The size of current action is: ', actions.shape, 'The size of current state is: ', states.shape)

        # Get target from the target network
        Q_target_next = self.target(next_state).detach().max(1)[0].unsqueeze(1)

        # Update the Q(s, a) using the Bellman Equation
        Q_next = rewards + (gamma * (1 - done) * Q_target_next)

        # Predict the Q from
        Q_predicted = self.actor(states).gather(1, actions)

        loss = F.mse_loss(Q_predicted, Q_next)

        # Update the network through back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.actor, self.target, TAU)

    def soft_update(self, actor, target, tau):
        """
        Q_target = tau * Q_actor + (1 - tau) * Q_target
        :param actor:
        :param target:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target.parameters(), actor.parameters()):
            target_param.data.copy_(tau * local_param + (1.0 - tau) * target_param)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """

        :param action_size: size of action space
        :param buffer_size: length of replay memory
        :param batch_size: length of minibatch
        :param seed:
        """
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # Sample a minibatch from experience buffer
        experiences = random.sample(self.memory, k=self.batch_size)
        # Convert the sampled experience to numpy tensor
        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
