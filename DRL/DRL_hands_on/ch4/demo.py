import numpy as np
import gym
from keras.optimizers import adam
from keras.models import model_from_json


ENV_GAME = "CartPole"
MODEL_ARCHITECTURE = './models/keras_{}.json'.format(ENV_GAME)
WEIGHTS_FILE = "./models/keras_{}.h5".format(ENV_GAME)

env = gym.make('{}-v0'.format(ENV_GAME))

# load json and create model
json_file = open(MODEL_ARCHITECTURE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(WEIGHTS_FILE)
print("Loaded model from disk")
 
# evaluate loaded model on test data
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr=0.001, decay=1e-6), metrics=['accuracy'])

print("===== Demo mode =====")
# play the game by the trained agent
total_reward = 0.0
total_steps = 0
obs = env.reset()
env.render()

while True:
	env.render()
	action = np.random.choice(env.action_space.n, p=model.predict(np.array([obs]))[0])
	obs, r, done, _ = env.step(action)
	total_reward += r
	total_steps += 1

	if done:
		break
print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))