from blackjack import BlackjackEnv, print_observation, strategy
import numpy as np

def First_Visit_MC():
	return state_value


if __name__ == '__main__':
	env = BlackjackEnv()
	state_value = dict()
	policy = np.ones([env.nA])/env.nA
	print(state_value, policy)

	# observe the environment and store the observation
	experience = []
	observation = env.reset()
	for t in range(100):
		action = strategy(observation)
		next_observation, reward, done, _ = env.step(action)
		experience.append((observation, action, reward))
		observation = next_observation
		if done:
			break
	print(experience)

	# update the state-value function using the obtained experience
	Returns = dict()
	for step in experience:
		observation, action, reward = step
		Returns[observation] = reward
