# check the answer
# https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb

from grid_world import GridworldEnv
import numpy as np

env = GridworldEnv()
initial_state = env.reset()
state_value = np.zeros(env.nS)
policy = np.ones([env.nS, env.nA])/env.nA
gamma = 0.5

print(state_value)
for i in range(3):
	for index in range(env.nS):
		v = state_value[index]
		for a in range(len(policy)):
			next_s, r, done, _ = env.step(a)
			state_value[index] = policy[a]*(r + gamma*state_value[next_s])
			print(next_s, r, done)
			env._render()


	next_s, r, done, _ = env.step(env.action_space.sample())
	print(next_s, r, done)
	env._render()