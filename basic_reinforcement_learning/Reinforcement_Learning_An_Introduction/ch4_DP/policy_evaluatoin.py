# Following the algo in section 4.1 Policy Evaluation (Prediction)

from grid_world import GridworldEnv
import numpy as np

def DP(env, policy, gamma, theta):
	while True:
		delta = 0
		for s in range(env.nS):
			v = 0
			for a in range(env.nA):
				p, next_s, r, _ = env.P[s][a][0]
				v += p*policy[s][a]*(r + gamma*state_value[next_s])
			delta = max(delta, np.abs(v - state_value[s]))
			state_value[s] = v
		if delta < theta:
			break
	return(np.array(state_value).reshape(env.shape))

env = GridworldEnv()
state_value = np.zeros(env.nS)
policy = np.ones([env.nS, env.nA])/env.nA
gamma = 1.0
theta = 0.00001

state_value = DP(env, policy, gamma, theta)

print("===== Training Finished =====")
print(state_value)


print("===== Play game =====")

s = env.reset()
for i in range(10):
	a = np.random.choice(env.nA, p=policy[s])
	next_s, r, done, _ = env.step(a)
	print()
	s = next_s
	env._render()