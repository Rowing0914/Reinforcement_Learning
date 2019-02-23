from time import time

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard
import numpy as np

x_train, y_train = np.random.rand(5,5), np.random.randint(2, size=5)
x_test, y_test = np.random.rand(3,5), np.random.randint(2, size=3)

model = Sequential()
model.add(Dense(3, input_dim=5, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='sgd', loss='mse')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[tensorboard])
score = model.evaluate(x_test, y_test)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])