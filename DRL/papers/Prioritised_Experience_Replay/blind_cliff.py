"""
Author: Norio Kosaka
E-mail: kosakaboat@gmail.com
Date: 20 Mar 2019
Description:
  This is the rough implementation of the Q-learnig agent on the Blind Cliffwalk environment.
  In the paper, Figure 1 describes the uniformly random agent linearly increases the number of updates needed, whrereas Q-learnign agent does not.
  So to reproduce the result, I have implemented the thing.

"""

import numpy as np
from collections import deque
from random import shuffle

class blind_cliff:
  """
  Blind Cliffwalk:
    there are two actions on this env(right or wrong)
  
  Game Logic:
    whenever the agent selects wrong action, this returns GAME OVER
    otherwise, it keep sliding the agent's position to one step right
    And at the most right position, the agent can clear the game and he will get a reward of 1 otherwise 0.
  
  Terminal State:
    either the agent reaches the most right state or it selects wrong action

  Sample Game
  Step:  0
  |*|| || || || || || || || || |
  Step:  1
  | ||*|| || || || || || || || |
  Step:  2
  | || ||*|| || || || || || || |
  Step:  3
  | || || ||*|| || || || || || |
  Step:  4
  | || || || ||*|| || || || || |
  Step:  5
  | || || || || ||*|| || || || |

  """
  def __init__(self, n):
    self.action_n  = 2    # number of action
    self.n         = n    # state space
    self.env       = ["| |"]*n
    self.agent_pos = 0
    self.env[self.agent_pos] = "|*|"

  def step(self, action):
    """
    Change the agent's position according to its action

    Args:
      action: the agent's action

    Returns:
      state: agent's position
      reward: 0(before the terminal state) or 1(at the terminal state)
      True/False: if game has ended
    """
    if self.agent_pos == self.n-1:
      # you have reached the terminal state
      return self._decode(), 1, True
    else:
      # if the action was `right`
      if action == 0:
        # move toward right
        self.env[self.agent_pos] = "| |"
        self.agent_pos += 1
        self.env[self.agent_pos] = "|*|"
      # if the action was `wrong`
      elif action == 1:
        # GAME OVer
        return self._decode(), 0, True
      # you have not reached the terminal state
      return self._decode(), 0, False

  def render(self):
    """
    Print out the env on the console
    """
    print("".join(self.env))

  def reset(self):
    """
    reset the agent position at 0
    """
    self.env       = ["| |"]*self.n
    self.agent_pos = 0
    self.env[self.agent_pos] = "|*|"
    return 0

  def _decode(self):
    """
    this decodes the string format of environemnt to the sequence of number
    which represents the agent position and other empty space by 1 and 0.

    Sample: assume that state space is 4
      |*|| || || |  ==> [1,0,0,0]
      | ||*|| || |  ==> [0,1,0,0]
      | || ||*|| |  ==> [0,0,1,0]
    """
    temp = np.zeros(self.n)
    temp[self.agent_pos] += 1
    return temp



class Agent:
  """
  Q-learning agent
  """
  def __init__(self, alpha, gamma, eta, epsilon, memory_size, batch_size, state_space, action_n, is_prioritised, Q=None):
    self.epsilon = epsilon
    self.alpha   = alpha
    self.gamma   = gamma
    self.memory  = deque(maxlen=memory_size) # EXPERIECNE REPLAY MEMORY
    self.weights = np.random.normal(0, 0.01, [state_space, action_n]) # gaussian distribution: mean 0 and sigma 0.01
    self.eta     = eta
    self.batch_size = batch_size
    self.is_prioritised = is_prioritised
    self.Q       = Q

  def __repr__(self):
    """
    format the output on the console when it is printed
    """
    return str(dict(self.Q))

  def policy(self, state):
    """
    Annealing Epsilon-greedy policy

    the agent reduce the epsilon over timestep
    and eventually it will only rely on its learned Q-table
    """
    if np.random.rand() > self.epsilon:
      action = np.argmax(self.function_approximator(state))
    else:
      action = np.random.choice([0,1], p=[0.5, 0.5])
    self.epsilon *= 0.99
    return action

  def function_approximator(self, state):
    """
    we use a very simple encoding of state as a 1-hot vector (as for tabular)

    Args:
      state: 1 x n vector(one hot vector) e.g., 
          Sample: assume that state space is 4
          |*|| || || |  ==> [1,0,0,0]
          | ||*|| || |  ==> [0,1,0,0]
          | || ||*|| |  ==> [0,0,1,0]

    Returns:
      Q-values: 1 x 2(number of action)
    """
    return np.dot(state, self.weights) + 1.0 # 1 is bias

  def update(self):
    """
    Update weights used in function approximation by SGD
    tricky thing is that the error we want to propagate is not MSE or usual thinsg in ML literature
    here we use the TD error to learn.
    """

    if self.is_prioritised:
      states, actions, rewards, next_states, dones = self._prioritise_sample()
    else:
      states, actions, rewards, next_states, dones = self._sample()
    
    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
      td_delta = reward + self.gamma*np.max(self.function_approximator(next_state)) - self.function_approximator(state)[action]
      self.weights += self.eta*td_delta*self.function_approximator(state)[action]

  def _sample(self):
    """
    randomly select a chunk of experiences to break the correlation in history(memory)
    """
    indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
    states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
    return states, actions, rewards, next_states, dones

  def _prioritise_sample(self):
    """
    select a chunk of experiences which only contains good experiences
    """

    def getKey(item):
      # sort the memory by rewards
      return item[2]
    states, actions, rewards, next_states, dones = zip(*sorted(list(self.memory), key=getKey, reverse=True)[:self.batch_size])
    return states, actions, rewards, next_states, dones

if __name__ == '__main__':
  EPSILON = 0.99
  ALPHA = 1
  ETA   = 1./4
  NUM_EPISODES = 100
  MEMORY_SIZE = 200
  BATCH_SIZE = 16
  FREQUENCY_UPDATE = 20
  logs_PER, logs_NONPER = list(), list()

  for n in range(2, 16):
    env = blind_cliff(n) # GAME ENV
    GAMMA = 1 - 1./n

    # Q-learning with Prioritised Experience Replay
    agent = Agent(ALPHA, GAMMA, ETA, EPSILON, MEMORY_SIZE, BATCH_SIZE, env.n, env.action_n, is_prioritised=True)
    rewards = 0

    for i in range(1,NUM_EPISODES):
      state = env.reset()
      while True:
        # if you want to see how the agent moves, open this!
        # env.render()

        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        agent.memory.append((state, action, reward, next_state, done))
        if done:
          rewards += reward
          break
        elif (i%FREQUENCY_UPDATE) == 0:
          agent.update()

        state = next_state
    logs_PER.append(rewards)

  for n in range(2, 16):
    env = blind_cliff(n) # GAME ENV
    GAMMA = 1 - 1./n

    # Q-learning without Prioritised Experience Replay
    agent = Agent(ALPHA, GAMMA, ETA, EPSILON, MEMORY_SIZE, BATCH_SIZE, env.n, env.action_n, is_prioritised=False)
    rewards = 0

    for i in range(1,NUM_EPISODES):
      state = env.reset()
      while True:
        # if you want to see how the agent moves, open this!
        # env.render()

        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        agent.memory.append((state, action, reward, next_state, done))

        if done:
          rewards += reward
          break
        elif (i%FREQUENCY_UPDATE) == 0:
          agent.update()

        state = next_state
    logs_NONPER.append(rewards)

import matplotlib.pyplot as plt
plt.plot(np.arange(len(logs_PER)), logs_PER, label="PER")
plt.plot(np.arange(len(logs_NONPER)), logs_NONPER, label="NON PER")
plt.legend()
plt.show()