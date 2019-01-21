import gym
import numpy as np
import pandas as pd

class Agent:
	def __init__(self, epsilon, num_action):
		# Q table Format: {state: {action: reward}}
		self.q = dict()
		self.epsilon = epsilon
		self.num_action = num_action
		self.policy  = self._initialise_policy()

	def _initialise_policy(self):
		# initialise a policy uniformly
		return np.ones(self.num_action)*(1/self.num_action)

	def learn(self, state, action, reward):
		self.q[str(state)] = {str(action):reward}
		# print(self.q[str(state)][str(action)])

	def choose_action(self, state, isTraining=True):
		if isTraining:
			return self._explore(state)
		else:
			if np.random.rand(1) > self.epsilon:
				# Exploitation
				return self._exploit(state)
			else:
				# Exploration
				return self._explore(state)

	def _exploit(self, state):
		return self.q.get(state)

	def _explore(self, state):
		return np.random.choice(np.arange(self.num_action), p=self.policy)

# Inspired by vmayoral
# https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/q-learning-gym-1.py#L74 
def build_state(observation):
	cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
	features = [_to_bin(cart_position, cart_position_bins),
				_to_bin(pole_angle, pole_angle_bins),
				_to_bin(cart_velocity, cart_velocity_bins),
				_to_bin(angle_rate_of_change, angle_rate_bins)]
	return int("".join(map(lambda feature: str(int(feature)), features)))

def _to_bin(value, bins):
	return np.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
	global cart_position_bins, pole_angle_bins, cart_velocity_bins, angle_rate_bins
	n_bins = 8
	n_bins_angle = 10
	cart_position_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
	pole_angle_bins = pd.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
	cart_velocity_bins = pd.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
	angle_rate_bins = pd.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

	env = gym.make('CartPole-v0')
	agent = Agent(epsilon=0.2, num_action=env.action_space.n)
	isTraining = True

	for episode in range(10):
		observation = env.reset()
		state = build_state(observation)
		for t in range(100):
			env.render()
			action = agent.choose_action(state, isTraining)
			observation, reward, done, info = env.step(action)
			state = build_state(observation)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
			else:
				print(state, action, reward)
				agent.learn(state, action, reward)