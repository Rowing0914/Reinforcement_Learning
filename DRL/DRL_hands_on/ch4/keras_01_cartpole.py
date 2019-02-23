# original paper of Cross-Entropy Method: https://link.springer.com/content/pdf/10.1007%2Fs10479-005-5732-z.pdf

from collections import namedtuple
import numpy as np
import gym
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import adam
from keras.callbacks import TensorBoard

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class Agent():
	def __init__(self, obs_size, hidden_size, n_actions):
		model = Sequential()
		model.add(Dense(hidden_size, input_dim=obs_size, activation='relu'))
		model.add(Dense(n_actions, input_dim=hidden_size, activation='softmax'))
		# since we are dealing with act(0 or 1), we should use sparse_categorical_crossentropy
		# or if we convert this act into one-hot vector, then we can use categorical_crossentropy for loss.
		model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
		model.summary()
		self.model = model
		self.n_actions = n_actions

	def _Q_values(self, obs):
		return self.model.predict(np.array([obs]))

	def choose_action(self, obs):
		# print(self._Q_values(obs)[0])
		return np.random.choice(self.n_actions, p=self._Q_values(obs)[0])


def iterate_batches(agent, env, batch_size):
	batch = []
	episode_reward = 0.0
	episode_steps = []
	obs = env.reset()
	while True:
		action = agent.choose_action(obs)
		next_obs, reward, is_done, _ = env.step(action)
		episode_steps.append(EpisodeStep(observation=obs, action=action))
		episode_reward += reward

		if is_done:
			batch.append(Episode(reward=episode_reward, steps=episode_steps))

			# initialise the memory
			episode_reward = 0.0
			episode_steps = []
			next_obs = env.reset()

			if len(batch) == batch_size:
				yield batch
				batch = []
		obs = next_obs


def filter_batch(batch, percentile):
	rewards = list(map(lambda s: s.reward, batch))
	reward_bound = np.percentile(rewards, percentile)
	reward_mean = float(np.mean(rewards))

	train_obs = []
	train_act = []
	for example in batch:
		if example.reward < reward_bound:
			continue
		train_obs.extend(map(lambda step: step.observation, example.steps))
		train_act.extend(map(lambda step: step.action, example.steps))

	return train_obs, train_act, reward_bound, reward_mean



if __name__ == '__main__':
	HIDDEN_SIZE = 128
	BATCH_SIZE = 16
	PERCENTILE = 70
	STOP = 199
	ENV_GAME = "CartPole"

	env = gym.make('{}-v0'.format(ENV_GAME))
	obs_size = env.observation_space.shape[0]
	n_actions = env.action_space.n

	agent = Agent(obs_size, HIDDEN_SIZE, n_actions)
	for iter_no, batch in enumerate(iterate_batches(agent, env, BATCH_SIZE)):
		obs, act, reward_b, reward_m = filter_batch(batch, PERCENTILE)

		# keras fit can update the parameters in on-line manner(https://stackoverflow.com/questions/50448743/is-it-logical-to-loop-on-model-fit-in-keras)
		history = agent.model.fit(np.array(obs), np.array(act), epochs=iter_no+1, initial_epoch=iter_no, shuffle=True, callbacks=[tbCallBack])
		print(history)
		if reward_m > STOP:
			print('Solved')
			# serialize model to JSON
			model_json = agent.model.to_json()
			with open("./models/keras_{}.json".format(ENV_GAME), "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			agent.model.save_weights("./models/keras_{}.h5".format(ENV_GAME))
			print("Saved model to disk")
			break

	print("===== Demo mode =====")
	# play the game by the trained agent
	total_reward = 0.0
	total_steps = 0
	obs = env.reset()
	env.render()

	while True:
		env.render()
		action = agent.choose_action(np.array(obs))
		obs, r, done, _ = env.step(action)
		total_reward += r
		total_steps += 1

		if done:
			break
	print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))