## Main Paper

Deterministic Policy Gradient Algorithms by D.Silver et al., IMCL2014



## Problem Statement

- The stochastic policy gradients may require more samples especially if the action space has many dimensions. Yet to guarantee the exploration of the env, we need the stochasticity of the policy.



## Proposition

- To ensure that our deterministic policy gradient algorithms continue exploring sufficiently, they introduce an off-policy learning algorithm of Actor-critic model. The basic idea is to select actions according to a stochastic behaviour policy, but to learn about a deterministic target policy.



## Experiments

- We apply our deterministic actor-critic algorithms to several benchmark problems:
  - a high-dimensional bandit; several standard benchmark reinforcement learning tasks with low dimensional action spaces;
  - a high-dimensional task for controlling an octopus arm.



## Conclusion



