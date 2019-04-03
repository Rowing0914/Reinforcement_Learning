## Main Paper

Deterministic Policy Gradient Algorithms by D.Silver et al., IMCL2014



## Problem Statement

- The stochastic policy gradients may require more samples especially if the action space has many dimensions. Yet to guarantee the exploration of the env, we need the stochasticity of the policy.

- Both REINFORCE and the vanilla version of actor-critic method are on-policy: training samples are collected according to the target policy — the very same policy that we try to optimize for. Off policy methods, however, result in several additional advantages:
  - The off-policy approach does not require full trajectories and can reuse any past episodes ([“experience replay”](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#deep-q-network)) for much better sample efficiency.
  - The sample collection follows a behavior policy different from the target policy, bringing better [exploration](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#exploration-exploitation-dilemma).

## Proposition

- To ensure that our deterministic policy gradient algorithms continue exploring sufficiently, they introduce an off-policy learning algorithm of Actor-critic model. The basic idea is to select actions according to a stochastic behaviour policy, but to learn about a deterministic target policy.



## Experiments

- We apply our deterministic actor-critic algorithms to several benchmark problems:
  - a high-dimensional bandit; several standard benchmark reinforcement learning tasks with low dimensional action spaces;
  - a high-dimensional task for controlling an octopus arm.



## Conclusion



## Reference

- <https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html>