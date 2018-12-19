# 5.3 Monte Carlo Control
# Monte Carlo ES (Exploring Starts) with first visit constraint

from collections import defaultdict
from blackjack import BlackjackEnv, print_observation
import numpy as np

def First_Visit_MC_ES(env, action_value, policy, discount_factor=1.0, num_episodes=1000):
	Returns = defaultdict(float)
	Returns_count = defaultdict(float)

	for i in range(num_episodes):
		# observe the environment and store the observation
		experience = []
		# this satisfies the exploraing start condition
		observation = env.reset()
		# generate an episode
		for t in range(100):
			action = policy(observation)
			next_observation, reward, done, _ = env.step(action)
			experience.append((observation, action, reward))
			observation = next_observation
			if done:
				break

		# remove duplicated state-action pairs in the episode
		state_action_in_experience = set([(x[0], x[1]) for x in experience])
		# update the state-value function using the obtained episode
		for row in state_action_in_experience:
			state, action = row[0], row[1]
			# Find the first occurance of the state-action pair in the episode
			first_occurence_idx = next(i for i,x in enumerate(experience) if ((x[0] == state) and (x[1] == action)))
			# Sum up all discounted rewards over time since the first occurance in the episode
			G = sum([x[2]*(discount_factor**i) for i,x in enumerate(experience[first_occurence_idx:])])
			# Calculate average return for this state over all sampled experiences
			Returns[row] += G
			Returns_count[row] += 1.0
			action_value[state][action] = Returns[row] / Returns_count[row]

	# remove duplicated states in the episode
	state_in_experience = set([(x[0]) for x in experience])
	# initialise the policy based on the stored observations ever
	policy = np.ones([len(state_in_experience), env.nA])/env.nA
	print(policy.shape)
	asdf
	# for state in state_in_experience:

	return action_value

def policy(observation):
	# Action => {0: "Stick", 1: "Hit"}
	return 0 if observation[0] >= 20 else 1

if __name__ == '__main__':
	env = BlackjackEnv()
	action_value = defaultdict(lambda: np.zeros(env.action_space.n))
	discount_factor = 1.0
	num_episodes = 100
	action_value = First_Visit_MC_ES(env, action_value, policy, discount_factor=1.0, num_episodes=num_episodes)
	print(action_value)