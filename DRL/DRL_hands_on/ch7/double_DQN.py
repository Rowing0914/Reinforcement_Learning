# to remove unnecessary path from python path
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import gym
import random
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, Dense
from keras.optimizers import Adam
from collections import deque
from lib import wrappers


EPISODES = 300
IMAGE_SIZE = (4,84,84)

DEFAULT_ENV_NAME = "BreakoutDeterministic-v4"
# DEFAULT_ENV_NAME = "PongNoFrameskip-v4"

class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # if self.load_model:
        #     self.model.load_weights("./save_model/dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, input_shape=IMAGE_SIZE, kernel_size=8, strides=4, activation='relu', data_format='channels_first'))
        model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size)) # calcualte Q-vaules for each action
        model.compile(loss='mse', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reshape_state(self, state):
        return state.reshape(1, 4, 84, 84)

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        state = self.reshape_state(state)
        q_value = self.model.predict(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), q_value[0]
        else:
            return np.argmax(q_value[0]), q_value[0]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _sample(self):
        """
        randomly select a chunk of experiences to break the correlation in history(memory)
        """
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        # sample experiences in memory
        states, actions, rewards, dones, next_states = self._sample()

        # Q values to specify the action at the state following the Q values computed by the target network
        base_state_values = self.model.predict_on_batch(next_states)

        # target Q-values
        next_state_values = self.target_model.predict_on_batch(next_states)

        # make next state values 0 at terminal state
        next_state_values[dones] = 0.0

        # calculate the one-step forward
        expected_state_action_values = next_state_values * self.discount_factor + np.array([rewards,]*self.action_size).T

        # fit() vs train_on_batch() in keras? => https://stackoverflow.com/questions/49100556/what-is-the-use-of-train-on-batch-in-keras
        # update Q-network using the tartet network's output
        # X: states, y: output of target network
        self.model.train_on_batch(states, expected_state_action_values)


if __name__ == "__main__":
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    # get size of state and action from environment
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes, actions = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action, q_value = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # if an action make the episode end, then gives penalty of -100
            # reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, done, next_state)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            # to check if it overestimates the Q_value
            actions.append(q_value)

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            # agent.model.save_weights("./save_model/dqn.h5")