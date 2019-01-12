## Introduction

This is the notes for the research paper below

* [A Comprehensive Survey on Safe Reinforcement Learning](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) by Javier Garcia and Fernando Fernandez in 2015



## To-do

- Robust MDPs
- Risk-Aware MDPs



## Papers

#### [Matthias Heger. Risk and reinforcement learning: concepts and dynamic programming. ZKW-Bericht. ZKW, 1994a.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.45.8264&rep=rep1&type=pdf)

- Abstract
  - it is well known that it is not always reliable and can be treacherous to use the expected value as a decision criterion [Tha 87]
  - A lot of alternative decision criteria have been suggested in decision theory to get a more sophisticated consideration of risk but most RL researchers have not concerned themselves with this subject until now.
  - The purpose of this paper is to draw the reader’s attention to the problems of the expected value criterion in Markov Decision Processes and to give Dynamic Programming algorithms for an alternative criterion, namely the Minimax criterion
- Proposal
  - A counterpart to Watkins' Q-learning, he proposed $\hat{Q}$-learning that finds policies that minimise the
    worst-case total discounted costs. 
- Experiments
  - no experiments has been done
- Conclusions
  - this paper discusses the problem of the criterion of the expected return as a measure of the performance for policies.

#### [Risk-sensitive and minimax control of discrete-time, finite-state Markov decision processes by Coraluppi and Marcus, 1997](https://ac.els-cdn.com/S0005109898001538/1-s2.0-S0005109898001538-main.pdf?_tid=41f12bb7-791e-407b-a134-108737a285c5&acdnat=1546876753_57a2f080a1d9373bd4e015770ee27734)

- Abstract
  - This paper analyzes a connection between risk-sensitive and minimax criteria for discrete-time, finite-state Markov decision processes (MDPs).
- Proposal
  - A generalized decision-making framework is introduced, which includes as special cases a number of approaches that have been considered in the literature. The framework allows for discounted risk-sensitive and minimax formulations leading to stationary optimal policies on the infinite horizon. 
  - Finite state space,,,
- Experiments
  - machine replacement example
- Conclusions
  - Key results include a large-risklimit connection between risk-sensitive and minimax control in the MDP setting, infinite horizon discounted dynamic programming equations for both risk-sensitive and minimax criteria, and a generalized framework for discounted optimal decision-making, allowing for controllers that retain risk-sensitivity without sacrificing stationarity on the infinite horizon.

#### [Minimax Reinforcement Learning by Chakravorty and Hyland in 2003](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/76233/AIAA-2003-5718-555.pdf?sequence=1&isAllowed=y)

- Abstract
  - In this paper, the minimax actor-critic algorithm is presented in dynamic programming setting. It is a deterministic worst case treatment of uncertainty, applied to a sequential decision-making problem.
- Proposal
  - minimax actor-critic algorithm with DP
  - this is a first ever research regarding the adaption of function approximation for minimax/worst case reinforcement learning.
- Experiments
  - UAV navigating experiment
- Conclusions
  - they have successfully adapted minimax attitude to Actor-critic algorithm.
  - They have presented the bounds on the errors that result from approximating a higher order system by a lower order system.

#### [Chris Gaskett. Reinforcement learning under circumstances beyond its control. In Proceedings of the International Conference on Computational Intelligence for Modelling Control and Automation, 2003.](https://pdfs.semanticscholar.org/7c61/0c97c56e9e3108af9ac00faacce5dbd2ac0c.pdf)

- Abstract
  - it provides robust decision-making criteria that support decision-making under conditions of uncertainty or risk. Decision theory has been applied to produce reinforcement learning algorithms that manage uncertainty in state-transitions. However, performance when there is uncertainty regarding the selection of future actions must also be considered. So He proposed the $\beta$-pessimistic Q-learning
- Proposal
  - $\beta$-pessimistic Q-learning
- Experiment
  - the cliff edge game
- Conclusions
  - $\beta$-pessimistic Q-learning is a promising algorithm which can find policies which are robust to the effects of nondeterministic action-selection. Unlike Sarsa, it is capable of off-policy learning. It is also computationally feasible, even for learning problems with continuous actions. The factor β allows adjustment between optimism and pessimism. Beyond the particular algorithm, considering nondeterministic action-selection leads to a robustl earning system that seeks decision policies tolerant of actions beyond its control.

#### [Drew Bagnell, Andrew Ng, and Jeff Schneider. Solving uncertain markov decision problems. Technical report, Robotics Institute Carnegie Mellon, 2001.](https://pdfs.semanticscholar.org/85c8/51b739b4c7fae13bc7554f34f0ceec00f510.pdf)

- Abstract
  - The authors consider the fundamental problem of finding good policies in uncertain models. It is demonstrated that although the general problem of finding the best policy with respect to the worst model is NP-hard, in the special case of a convex uncertainty set the problem is tractable.The authors demonstrate that the uncertain model approach can be used to solve a class of nearly Markovian Decision Problems, providing lower bounds on performance in stochastic models with higher-order interactions.
- Proposal
  - Robust Value Iteration in the dynamic game, 
- Experiments
  - path planning for the dynamic object tracking task
  - Mountain Car POMDP
- Conclusions
  - The authors in this work have considered diverse application of the stochastic robustness framework including bounding error due to discretization, guaranteeing performance in reinforcement learning problems, and robust planning for a difficult to model dynamic obstacle
  - Future work may consider the application to variable resolution discretization in optimal control, as the technique naturally provides error bounds on the discretization. It would also be interesting to apply the technique to the kind of \assumed-density planning" demonstrated as practical POMDP solution algorithm by [Roy and Thrun, 1999] and [Rodriquez et al., 1999] to achieve a measure of robustness to the errors introduced by the reduced belief-state.

#### [Robust Markov Decision Processes for Medical 2 Treatment Decisions. Yuanhui Zhang et al., 2015](http://www.optimization-online.org/DB_FILE/2015/10/5134.pdf)

- Abstract
  - Medical treatment decisions involve complex tradeoffs between the risks and benefits of various treatment options. The diversity of treatment options that patients can choose over time and uncertainties in future health outcomes, result in a difficult sequential decision making problem. Markov decision processes (MDPs) are commonly used to study medical treatment decisions; however, optimal policies obtained by solving MDPs may be affected by the uncertain nature of the model parameter estimates.
  - In this article, we present a robust Markov decision process treatment model (RMDP-TM) with an uncertainty set that incorporates an uncertainty budget into the formulation for the transition probability matrices (TPMs) of the underlying Markov chain.
- Proposal
  - a robust Markov decision process treatment model (RMDP-TM) with an uncertainty set that incorporates an uncertainty budget into the formulation for the transition probability matrices (TPMs) of the underlying Markov chain.
- Experiments
  - we present an application of the models to a medical treatment decision problem of optimizing the sequence and the start time to initiate medications for glycemic control for patients with type 2 diabetes.
- Conclusions
  - We presented an RMDP model that can fit a broad range of medical treatment decisions in which there is uncertainty in transition probabilities. The interval model version of RMDP-TM model can be solved efficiently. 

#### [Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain transition matrices. Operational Research, 53(5):780–798, September 2005. ISSN 0030-364X.](https://people.eecs.berkeley.edu/~elghaoui/Pubs/RobMDP_OR2005.pdf)

- Abstract
  - Optimal solutions to Markov decision problems may be very sensitive with respect to the state transition probabilities. In many practical problems, the estimation of these probabilities is far from accurate. Hence, estimation errors are limiting factors in applying Markov decision processes to real-world problems.
  - We show that a particular choice of the uncertainty sets, involving likelihood regions or entropy bounds, leads to both a statistically accurate representation of uncertainty, and a complexity of the robust recursion that is almost the same as that of the classical recursion. Hence, robustness can be added at practically no extra computing cost. We derive similar results for other uncertainty sets, including one with a finite number of possible values for the transition matrices.
- Proposal
- Experiments
  - We describe in a practical path planning example the benefits of using a robust strategy instead of the classical optimal strategy; even if the uncertainty level is only crudely guessed, the robust strategy yields a much better worst-case expected travel time.
- Conclusions
  - they have examined numerous transition models and confirmed that some existing solutions work well on them.

#### [Aviv Tamar, Huan Xu, and Shie Mannor. Scaling Up Robust MDPs by Reinforcement Learning. Computing Research Repository, abs/1306.6189, 2013](https://arxiv.org/pdf/1306.6189.pdf)

- Abstract
  - Previous studies showed that robust MDPs, based on a minimax approach to handle uncertainty, can be solved using dynamic programming for small to medium sized problems. So they have extended it to the large scale problems by introducing their approach called *a robust approximate dynamic programming* based on a projected fixed point equation to approximately solve large scale robust MDPs.
- Proposal
  - a robust approximate DP
- Experiments
  - We show that the proposed method provably succeeds under certain technical conditions, and demonstrate its effectiveness through simulation of an option pricing problem.
- Conclusions
  - This work presented a novel framework for solving large-scale robust Markov decision processes. , such problems are beyond the capabilities of previous studies, which focused on exact solutions and hence suffer from the “curse of dimensionality”. Our approach to tackling the planning problem is through reinforcement learning methods: we reduce the dimensionality of the robust value function using linear function approximation, and employ an iterative sampling based procedure to learn the approximation weights. We presented both formal guarantees and empirical evidence to the usefulness of our approach in general robust MDPs, and optimal stopping problems in particular.

#### [Kurt Driessens and Saso Dzeroski. Integrating guidance into relational reinforcement learning. Machine Learning, 57(3):271–304, December 2004. ISSN 0885-6125](http://kt.ijs.si/SasoDzeroski/pdfs/2004/2004-DriessensEtAl-ML.pdf)

- Abstract
  - two problems in conventional RL approaches.
    - First, learning the Q-function in tabular form may be infeasible because of the excessive amount of memory needed to store the table, and because the Q-function only converges after each state has been visited multiple times
    - Second, rewards in the state space may be so sparse that with random exploration they will only be discovered extremely slowly
  - The first problem is often solved by learning a generalization of the encountered examples (e.g., using a neural net or decision tree) But, second one has not been addressed yet. The problem of sparse rewards has not been addressed for RRL. So, This paper presents a solution based on the use of “reasonable policies” to provide guidance. Different types of policies and different strategies to supply guidance through these policies are discussed and evaluated experimentally in several relational domains to show the merits of the approach.
- Proposal
- Experiments
- Conclusions

