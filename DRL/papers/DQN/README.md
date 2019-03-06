## Previous works

- Temporal DiWerence Learning and TD-Gammon  by Gerald Terauso
- Why did TD-Gammon Work? by Jordan B. Pollack & Alan D. Blair
- An Analysis of Temporal-Difference Learning with Function Approximation by John N. Tsitsiklis, Member, IEEE, and Benjamin Van Roy
- Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method by Martin Riedmiller



## Main Paper

- Playing Atari with Deep Reinforcement Learning by Volodymyr Mnih Koray Kavukcuoglu David Silver Alex Graves Ioannis Antonoglou Daan Wierstra Martin Riedmiller
- Human-level control through deep reinforcement learning by Mnih et al



## Summary

### the problem of Deep Learning:

1. delay between actions and resulting rewards
2. most DL model assumes IID assumption of data distribution: non-stationary and correlated dataset!!

### Goal and advantages of the proposition

- The goal is to create a single neural network agent that is able to successfully learn to play as many of the games as possible without having a game-specific architecture or weights.

- and the agent has to be able to operate directly on RGB images and efficiently process training data by using SGD.

- Abovementioned problems could be solved by using Experience replay mechanism.

### Deep Q-Network with Experience replay has several advantages;

1. each step of experience is potentially used in many weight updates => data efficiency
2. bootstrapping the experiences in memory helps to break the correlation of consecutive timesteps, therefore it reduces the variance of the updates
3. On-policy learning hugely biases the learning because the current parameters determine the next data sample that the parameters are trained on. It is easy to see how unwanted feedback loops may arise and the parameters could get stuck in a poor local minimum. Hence, using experience replay the behaviour distribution is averaged over many of its previous states,
4. smoothing out learning and avoiding oscillations or divergence in the parameters.

### Preprocessing and model architecture

- Raw images are 120 x 160 pixel RGB with a 128 colour palette
- but the processing raw images is computationally expensive so that they converted the raw images into gray-scale 110x84 images.
- then cropping an 84x84 region of the image that roughly captures the game screen from the resulting images
- the neural network architecture is
  - input: 84x84x4 image
  - 1st convolution layer: 16 8x8 filters with stride 4 followed by ReLU
  - 2nd convolution layer: 32 4x4 filters with stride 2 followed by ReLU
  - Dense(fully connected layer) followed by ReLU: **256** neurons exist
  - Dense(fully connected layer) followed by Softmax: **output: Q-values for each possible action in the game**
  - Optimiser: RMSProp
  - The selected games in Atari 2600: Beam Rider, Breakout, Enduro, Pong Q*bert, Seaquest, Space Invaders
  - Clipping rewards for 0, 1, -1.
  - Minibatch size: 32
  - Linearly Annealing **Epsilon** from 1 to a minimum of 0.1 over the first **1,000,000** steps, and fixed at **0.1** thereafter
  - Replay memory size: **1,000,000**
  - **Frame skipping-technique:** the agent only operates on every k-th frame instead of every frame and its last action repeated on next k-th frame as well where they set k 4 or 3.



### Algorithm

![dqn](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/DQN/images/DQN_algo.png)

### Result



![result](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/DQN/images/result.png)

![result](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/DQN/images/table.png)