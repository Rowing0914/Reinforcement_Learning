# to remove unnecessary path from python path
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import gym, time
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
from keras.optimizers import Adam
from collections import deque
from lib import wrappers

GAMMA = 0.99
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32
MEAN_REWARD_BOUND = 19.5
LEARNING_RATE = 0.005
TAU = .125
SYNC_TARGET_FRAMES = 1000

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPSILON_DECAY_LAST_FRAME = 10**5


DEFAULT_ENV_NAME = "BreakoutDeterministic-v4"
# DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

class DQN:
	def __init__(self, env):
		self.env           = env
		self.memory        = deque(maxlen=REPLAY_SIZE)
		self.gamma         = GAMMA
		self.epsilon       = EPSILON
		self.epsilon_min   = EPSILON_MIN
		self.epsilon_decay = EPSILON_DECAY
		self.learning_rate = LEARNING_RATE
		self.tau           = TAU
		
		self._create_model()

	def _create_model(self):
		model = Sequential()
		model.add(Conv2D(32, input_shape=(4,84,84), kernel_size=8, strides=4, activation='relu', data_format='channels_first'))
		model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
		model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.env.action_space.n))
		model.compile(loss='mse', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
		model.summary()

		self.model        = model
		self.target_model = clone_model(model)

	def epsilon_greedy(self, state):
		"""
		epsilon greedy policy to select an action at a state
		"""
		self._epsilon_decay

		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.model.predict(state))

	def _epsilon_decay(self):
		"""
		update epsilon
		"""
		self.epsilon = max(self.epsilon_min, EPSILON - frame_idx / EPSILON_DECAY_LAST_FRAME)

	def _sample(self):
		"""
		randomly select a chunk of experiences to break the correlation in history(memory)
		"""
		indices = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
		states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

	def replay(self):
		"""
		With randomly selected samples, we update the Q-Network using the target Q-values
		"""
		states, actions, rewards, dones, next_states = self._sample()

		# target Q-values
		next_state_values = self.target_model.predict_on_batch(next_states)
		
		# make next state values 0 at terminal state
		next_state_values[dones] = 0.0
		
		# calculate the one-step forward
		expected_state_action_values = next_state_values * GAMMA + np.array([rewards,]*self.env.action_space.n).T

		# fit() vs train_on_batch() in keras? => https://stackoverflow.com/questions/49100556/what-is-the-use-of-train-on-batch-in-keras
		# update Q-network using the tartet network's output
		# X = states, y = output of target network
		self.model.train_on_batch(states, expected_state_action_values)
		

	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)

	def save_model(self, fn):
		self.model.save(fn)

def logger(total_rewards, best_mean_reward, frame_idx, ts_frame, ts):
	speed = (frame_idx - ts_frame) / (time.time() - ts)
	ts_frame = frame_idx
	ts = time.time()
	mean_reward = np.mean(total_rewards[-100:])
	print("%d: done %d games, mean reward %.3f, speed %.2f f/s" % (
		frame_idx, len(total_rewards), mean_reward, speed
	))
	if best_mean_reward is None or best_mean_reward < mean_reward:
		best_mean_reward = mean_reward
	return best_mean_reward, mean_reward, frame_idx, ts_frame, ts

if __name__ == "__main__":
	env = wrappers.make_env(DEFAULT_ENV_NAME)
	total_rewards = []
	frame_idx = 0
	ts_frame = 0
	ts = time.time()
	best_mean_reward = None

	dqn_agent = DQN(env=env)
	state = env.reset()
	while True:
		frame_idx += 1
		action = dqn_agent.epsilon_greedy(state)
		new_state, reward, done, _ = env.step(action)
		dqn_agent.memory.append([state, action, reward, done, new_state])
		total_rewards.append(reward)

		if done:
			state = env.reset()

		# ======== LOG FUNCTIONS START ========
			best_mean_reward, mean_reward, frame_idx, ts_frame, ts = logger(total_rewards, best_mean_reward, frame_idx, ts_frame, ts)
			if mean_reward > MEAN_REWARD_BOUND:
				print("Solved in %d frames!" % frame_idx)
				break
		# ======== LOG FUNCTIONS END ========

		if (frame_idx % SYNC_TARGET_FRAMES == 0) and (frame_idx > SYNC_TARGET_FRAMES):
			# Update Target Network
			dqn_agent.target_train()
		elif len(dqn_agent.memory) >= REPLAY_START_SIZE:
			# Update Q-Network
			dqn_agent.replay()
			# dqn_agent.save_model("success.model")
		
		state = new_state