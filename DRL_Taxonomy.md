## Reference

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8632747&tag=1



## DRL approaches Taxonomy

### Arcade Games

- DQN
  - DQN was tested in seven Atari 2600 games and outperformed previous approaches, such as SARSA with feature construction as well as a human expert on three of the games.
- Deep Recurrent Q-Learning (DRQN)
  - extends the DQN architecture with a recurrent layer before the output and works well for games with partially observable states
- Gorila architecture (General Reinforcement Learning Architecture)
  - A distributed version of DQN was shown to outperform a non-distributed version in 41 of the 49 games 
  - Gorila parallelizes actors that collect experiences into a distributed replay memory as well as parallelizing learners that train on samples from the same replay memory.
- Double DQN
  - reduces the observed overestimation by learning two value networks with parameters that both use the other network for value-estimation
- prioritized experience replay
  - important experiences are sampled more frequently based on the TD-error, which was shown to significantly improve both DQN and Double DQN
- Dueling DQN
  - uses a network that is split into two streams after the convolutional layers to separately estimate state-value and the action-advantage functions.
  - Dueling DQN improves Double DQN and can also be combined with prioritized experience replay
- Bootstrapped DQN
  - improves exploration by training multiple Q-networks. A randomly sampled network is used during each training episode and bootstrap masks modulate the gradients to train the networks differentlly
- Asynchronous Advantage Actor-Critic (A3C)
  - an actor-critic method that uses several parallel agents to collect experiences that all asynchronously update a global actor-critic network without **replay memory**
  - A3C outperformed Prioritized Dueling DQN, which was trained for 8 days on a GPU, with just half the training time on a CPU
- actor-critic method with experience replay (ACER)
  - implements an efficient trust region policy method that forces updates to not deviate far from a running average of past policies
  - It is much more data efficient
- Advantage Actor-Critic (A2C)
  - a synchronous variant of A3C 
  - updates the parameters synchronously in batches and has comparable performance while only maintaining one neural network
- Actor-Critic using Kronecker-Factored Trust Region (ACKTR)
  - extends A2C by approximating the natural policy gradient updates for both the actor and the critic
- Trust Region Policy Optimization (TRPO) 
  - uses a surrogate objective with theoretical guarantees for monotonic policy improvement, while it practically implements an approximation called trust region by constraining network updates with a bound on the KL divergence between the current and the updated policy.
  - robust and data efficient performance in Atari games while it has high memory requirements and several restrictions
- Proximal Policy Optimization (PPO)
  - an improvement on TRPO that uses a similar surrogate objective but instead uses a soft constraint by adding the KL-divergence as a penalty
  - while it does not rely on **replay memory,** it has comparable or better performance than TRPO in continuous control tasks.
- IMPALA (Importance Weighted Actor-Learner Architecture)
  - an actor-critic method where multiple learners with GPU access share gradients between each other while being synchronously updated from a set of actors
- UNREAL (UNsupervised REinforcement and Auxiliary Learning)
  - based on A3C but uses a replay memory from which it learns auxiliary tasks and pseudo-reward functions concurrently
- [Distributional DQN](https://arxiv.org/pdf/1707.06887.pdf)
  - takes a distributional perspective on reinforcement learning by treating Q-function as an approximate distribution of returns instead of a single approximate expectation for each action as it is in the conventional setting.
  - The distribution is divided into a so-called set of atoms, which determines the granularity of the distribution.