"""
Author: Norio Kosaka
E-mail: kosakaboat@gmail.com
Date: 20 Mar 2019
Description:
  This is the rough implementation of the Q-learnig agent on the Blind Cliffwalk environment.
  In the paper, Figure 1 describes the uniformly random agent linearly increases the number of updates needed, whrereas Q-learnign agent does not.
  So to confirm, if it is true, I have implemented the thing.

"""

import numpy as np
from collections import defaultdict, deque

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
  def __init__(self, Q, alpha, gamma, eta, epsilon, memory_size, state_space, action_n):
    self.epsilon = epsilon
    self.Q       = Q
    self.alpha   = alpha
    self.gamma   = gamma
    self.memory  = deque(maxlen=memory_size) # EXPERIECNE REPLAY MEMORY
    self.weights = np.random.normal(0, 0.01, [state_space, action_n]) # gaussian distribution: mean 0 and sigma 0.01
    self.eta     = eta

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

  def update(self, state, action, reward, next_state, done):
    """
    Update weights used in function approximation by SGD
    tricky thing is that the error we want to propagate is not MSE or usual thinsg in ML literature
    here we use the TD error to learn.
    """
    td_delta = reward + self.gamma*np.max(self.function_approximator(next_state)) - self.function_approximator(state)[action]
    self.weights += self.eta*td_delta*self.function_approximator(state)[action]

  def _memory_maintainance(self):
    return

  def old_update(self, state, action, reward, next_state, done):
    """
    Update the Q-table
    """
    td_delta = reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action]
    self.Q[state][action] += self.alpha*td_delta

if __name__ == '__main__':
  EPSILON = 0.99
  ALPHA = 1
  ETA   = 1./4
  NUM_EPISODES = 100
  MEMORY_SIZE = 200
  logs = list()

  for n in range(2, 16):
    env = blind_cliff(n) # GAME ENV
    GAMMA = 1 - 1./n
    Q = defaultdict(lambda: np.random.normal(0, 0.01, env.action_n)) # Q TABLE
    agent = Agent(Q, ALPHA, GAMMA, ETA, EPSILON, MEMORY_SIZE, env.n, env.action_n) # Q-LEARNING AGENT
    rewards = 0

    for i in range(NUM_EPISODES):
      state = env.reset()
      while True:
        # if you want to see how the agent moves, open this!
        # env.render()

        action = agent.policy(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        if done:
          rewards += reward
          break
        state = next_state
    logs.append(rewards)

import matplotlib.pyplot as plt
plt.plot(logs)  
plt.show()