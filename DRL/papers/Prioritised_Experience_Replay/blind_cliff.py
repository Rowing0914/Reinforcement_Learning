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
from collections import defaultdict

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
    self.action_n  = 2
    self.n         = n
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
      return 0, 1, True
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
        return 0, 0, True
      # you have not reached the terminal state
      return self.agent_pos, 0, False

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

class Agent:
  """
  Q-learning agent
  """
  def __init__(self, Q, alpha, gamma, epsilon):
    self.epsilon = epsilon
    self.Q       = Q
    self.alpha   = alpha
    self.gamma   = gamma

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
      action = np.argmax(self.Q[state])
    else:
      action = np.random.choice([0,1], p=[0.5, 0.5])
    self.epsilon *= 0.99
    return action

  def update(self, state, action, reward, next_state):
    """
    Update the Q-table
    """
    td_delta = reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action]
    self.Q[state][action] += self.alpha*td_delta

if __name__ == '__main__':
  NUM_STATES = 10
  EPSILON = 0.99
  GAMMA = 1
  ALPHA = 0.5
  NUM_EPISODES = 100

  env = blind_cliff(NUM_STATES)
  Q = defaultdict(lambda: np.zeros(env.action_n))
  agent = Agent(Q, ALPHA, GAMMA, EPSILON)
  rewards = 0

  for i in range(NUM_EPISODES):
    print("===== GAME: {} =====".format(i))
    state = env.reset()
    while True:
      env.render()
      action = agent.policy(state)
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, next_state)
      if done:
        rewards += reward
        break
      state = next_state

print(agent)
print(rewards)