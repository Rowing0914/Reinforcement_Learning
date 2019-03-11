## Reference

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8632747&tag=1



## DRL approaches Taxonomy

### Arcade Games

- [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

  - DQN was tested in seven Atari 2600 games and outperformed previous approaches, such as SARSA with feature construction as well as a human expert on three of the games.

    <img src="images/DQN.png" width=60% height=60%>

- [Deep Recurrent Q-Learning (DRQN)](https://arxiv.org/pdf/1507.06527.pdf)

  - extends the DQN architecture with a recurrent layer before the output and works well for games with partially observable states

    <img src="images/DRQN.png" width=40% height=40%>

- [Gorila architecture (General Reinforcement Learning Architecture)](https://arxiv.org/pdf/1507.04296.pdf)

  - A distributed version of DQN was shown to outperform a non-distributed version in 41 of the 49 games 

  - Gorila parallelizes actors that collect experiences into a distributed replay memory as well as parallelizing learners that train on samples from the same replay memory.

    <img src="images/GORILA.png" width=60%>

- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)

  - reduces the observed overestimation by learning two value networks with parameters that both use the other network for value-estimation

- [prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf)

  - important experiences are sampled more frequently based on the TD-error, which was shown to significantly improve both DQN and Double DQN

    <img src="images/prioritised_experience_replay.png" width=60%>

- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)

  - uses a network that is split into two streams after the convolutional layers to separately estimate state-value and the action-advantage functions.

  - The main benefit of this factoring is to generalise learning across actions without imposing any change to the underlying reinforcement learning algorithm. 

  - Dueling DQN improves Double DQN and can also be combined with prioritized experience replay

    <img src="images/Duelling_DQN.png" width=60%>

- [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621.pdf)

  - improves exploration by training multiple Q-networks. A randomly sampled network is used during each training episode and bootstrap masks modulate the gradients to train the networks differentlly

    <img src="images/Bootstrapped_DQN.png" width=60%>

- [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)

  - an actor-critic method that uses several parallel agents to collect experiences that all asynchronously update a global actor-critic network without **replay memory**

  - A3C outperformed Prioritized Dueling DQN, which was trained for 8 days on a GPU, with just half the training time on a CPU

    <img src="images/A3C.png" width=60%>

- [actor-critic method with experience replay (ACER)](https://arxiv.org/pdf/1611.01224.pdf)

  - implements an efficient trust region policy method that forces updates to not deviate far from a running average of past policies
  - It is much more data efficient

- [Advantage Actor-Critic (A2C)](Asynchronous Methods for Deep Reinforcement Learning)

  - a synchronous variant of A3C 
  - updates the parameters synchronously in batches and has comparable performance while only maintaining one neural network

- [Actor-Critic using Kronecker-Factored Trust Region (ACKTR)](Scalable trust-region method for deep reinforcement
  learning using Kronecker-factored approximation)

  - extends A2C by approximating the natural policy gradient updates for both the actor and the critic

- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)

  - uses a surrogate objective with theoretical guarantees for monotonic policy improvement, while it practically implements an approximation called trust region by constraining network updates with a bound on the KL divergence between the current and the updated policy.
  - robust and data efficient performance in Atari games while it has high memory requirements and several restrictions

- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

  - an improvement on TRPO that uses a similar surrogate objective but instead uses a soft constraint by adding the KL-divergence as a penalty

  - while it does not rely on **replay memory,** it has comparable or better performance than TRPO in continuous control tasks.

    <img src="images/PPO.png" width=60%>

- [IMPALA (Importance Weighted Actor-Learner Architecture)](https://arxiv.org/pdf/1802.01561.pdf)

  - an actor-critic method where multiple learners with GPU access share gradients between each other while being synchronously updated from a set of actors

- [UNREAL (UNsupervised REinforcement and Auxiliary Learning)](https://arxiv.org/pdf/1611.05397.pdf)

  - based on A3C but uses a replay memory from which it learns auxiliary tasks and pseudo-reward functions concurrently

    <img src="images/UNREAL.png" width=60%>

- [Distributional DQN](https://arxiv.org/pdf/1707.06887.pdf)
  - takes a distributional perspective on reinforcement learning by treating Q-function as an approximate distribution of returns instead of a single approximate expectation for each action as it is in the conventional setting.

  - The distribution is divided into a so-called set of atoms, which determines the granularity of the distribution.

    <img src="images/distributional_DQN.png" width=60%>