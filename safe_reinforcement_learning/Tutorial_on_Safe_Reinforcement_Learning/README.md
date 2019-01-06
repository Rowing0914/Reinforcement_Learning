## Profile

- Title: [Tutorial on Safe Reinforcement Learning](https://las.inf.ethz.ch/files/ewrl18_SafeRL_tutorial.pdf)
- Authors: Felix Berkenkamp, Andreas Krause
- Organisation: ETH Zurich
- Venue: @EWRL, October 1 2018



## To do

#### Temporal Logic

- temporal logic tutorial in japanese
  http://hagi.is.s.u-tokyo.ac.jp/pub/staff/hagiya/kougiroku/jpf/modal-temporal.pdf
  http://www.cs.tsukuba.ac.jp/~mizutani/under_grad/programtheory/2014/2014-09.pdf
- temporal logic course in japanese
  https://ist.ksc.kwansei.ac.jp/~ktaka/ML/ML.html
- temporal logic and rl
  https://arxiv.org/pdf/1612.03471.pdf
  https://arxiv.org/pdf/1709.09611.pdf
- check CS 333: Safe and Interactive Robotics](https://dorsa.fyi/cs333/), and http://iliad.stanford.edu/
- investigate her research, [Dorsa Sadigh](https://dorsa.fyi/)

#### CMDP(Constrained MDP, Altman, 1999)

- this forms the environment for safe RL accounting for the constraints (safety conditions)

#### Model Predictive Control(MPC)

https://myenigma.hatenablog.com/entry/2016/07/25/214014

#### Control theory(includes major theories, MPC, region of attraction and Lyapunov functions)

http://people.ee.ethz.ch/~apnoco/Lectures2018/NLSC_lecture_notes_2018.pdf





## Notes

### Safety Definitions

#### Specifying safety requirements and quantify risk

- Examples(When do we need to consider the safety?) *details in Summary of reference
  - Therapeutic Spinal Cord Stimulation
    - Safe Exploration for Optimization with Gaussian Processes by Y. Sui, A. Gotovos, J. W. Burdick, A. Krause
    - Stagewise Safe Bayesian Optimization with Gaussian Processes by Y. Sui, V. Zhuang, J. W. Burdick, Y. Yue
  - Safe Controller Tuning
    - Safe Controller Optimization for Quadrotors with Gaussian Processes by F. Berkenkamp, A. P. Schoellig, A. Krause, ICRA 2016

#### Safety Criterion

** have a look at **Bayesian optimisation** and **temporal logic** beforehand

- specifying safety behaviour(safety is the similar concept as avoiding bad trajectories)
  - $g(\{ s_t, a_t \}^N_{t=0} ) = g(\tau) > 0$ : Monitoring temporal properties of continuous signals by O. Maler, D. Nickovic, FT, 2004
  - $g(\tau) = \min_{t=1:N} \Delta(s_t, a_t)$ : Safe Control under Uncertainty by D. Sadigh, A. Kapoor, RSS, 2016
- But the expected safety is not perfect rather misleading indication, since it averages the bad and good trajectories. So, we need to consider the range of the distribution of trajectories, e.g., the variance.
- Notion of safety
  - Expected risk: $E[G]$
  - Moment penalised: $E[e^{\tau G}]$
  - Value at Risk: $\text{VaR}_{\delta}[G] = \inf \{ \epsilon \in R: p(G \leq \epsilon) \} \geq \delta $
  - Conditional Value at Risk: $\text{CVaR}_{\delta}[G] = \frac{1}{\delta} \int^{\delta}_0 \text{VaR}_{\alpha}[G] d\alpha$
  - Worst-case: $g(\tau) > 0 , \ \forall_{\tau} \in \Gamma$
    - Robust Control
    - Formal Verification

- Acting safely in known environments

  - References

    - *[Negative Dynamic Programming](https://projecteuclid.org/download/pdf_1/euclid.aoms/1177699369),  [Lyapunov function](https://en.wikipedia.org/wiki/Lyapunov_function)* : Constrained Markov decision processes by Eitan Altman, CRC Press, 1999
    - Essentials of robust control by Kemin Zhou, John C. Doyle, PH, 1998
    - Robust control of Markov decision processes with uncertain transition matrices by Arnab Nilim, Laurent El Ghaoui, OR, 2005

  - **Imitation learning algorithms**

    - Data Aggregation
      $$
      \text{Generate state sequence with policy } \pi(s_t, \theta)\\
      D \leftarrow D \cup \{ (s-t, \pi^*(s_t)) \}^T_{t=1}\\
      \theta = \text{argmin}_{\theta} \Sigma_{s, a \in D} || \pi(s, \theta) - a_t||
      $$

      - Search-based Structured Prediction by Hal Daume III, John Langford, Daniel Marcu, ML, 2009
      - Efficient Reductions for Imitation Learning by Stephan Ross, Drew Bagnell, AISTATS 2010

    - Policy Aggregation
      $$
      \pi_0 = \pi^*\\
      \text{Generate state sequence } D \text{ with policy } \pi_i\\
      \theta_{i+1} = \alpha_0 \pi^*(s) + \Sigma^{i+1}_{j=1} \alpha_j \pi(s, \theta_j)
      $$

      - A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning by Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, 2011

  - **Safe Imitation learning** (Concept of safety in imitation learning)

    - $ || \pi(s_t, \theta) - \pi^*(s_t) || \leq \epsilon$ : Query-Efficient Imitation Learning for End-to-End Autonomous Driving by Jiakai Zhang, Kyunghyun Cho, AAAI, 2017
    - $Var[\pi(s_t, \theta)] \leq \gamma$ : EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning by Kunal Menda, Katherine Driggs-Campbell, Mykel J. Kochenderfer, arXiv2018

  - **Prior Knowledge as backup for learning**: to hold the safety at the early stage of learning process but the thing is that Need to know what is unsafe in advance. Without learning, need significant prior knowledge. The learner does not know what’s happening!

    - Provably safe and robust learning-based model predictive control by A. Aswani, H. Gonzalez, S.S. Satry, C.Tomlin, Automatica, 2013
    - Safe Reinforcement Learning via Shielding by M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Nickum, U. Topcu, AAAI, 2018
    - Linear Model Predictive Safety Certification for Learning-based Control by K.P. Wabersich, M.N. Zeilinger, CDC, 2018
    - Safe Exploration of State and Action Spaces in Reinforcement Learning by J. Garcia, F. Fernandez, JAIR, 2012
    - Safe Exploration in Continuous Action Spaces by G. Dalai, K. Dvijotham, M. Veccerik, T. Hester, C. Paduraru, Y. Tassa, arXiv, 2018

  - **Overview of expected safety pipeline**

    Using the sampled trajectory data, we divide it into two parts, a training set and test set. Subsequently, we apply CPO(Constrained Policy Optimisation by Joshua Achiam at el,. 2017) to the training stage and obtain a candidate policy, which is supposed to be examined on the prepared test set. Then combine the two policies aforementioned to safely operate the robot.

    - High Confidence Policy Improvement by Philip S. Thomas, Georgios Theocharous, Mohammad Havamzadeh, ICML 2015
    - Safe and efficient off-policy reinforcement learning by Remi Munos, Thomas Stepleton, Anna Harutyunyan, Marc G. Bellemare, NIPS, 2016
    - Constrained Policy Optimisation by Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel, ICML, 2017

### Summary of Safety Definitions

- Reviewed safety definitions
  - Stochastic: expected risk, moment penalised, VaR/CVaR
  - Worst-case: formal verification, robust control
- Confirmed how to safely initialise the prior knowledge
- Reviewed a first method for safe learning in expectation

-----

### Explicit Safe Exploration

- Model-free approaches
- Model-based approaches

In safe RL domain, broadly speaking, we have two categories, which is Model-free and Model-based approaches. In model-free approaches, we care the estimate of $J(\theta)$ and optimise it, whereas in model-based approaches we aim to estimate/identify the dangerous states then plan/control the agent. So firstly, we will look at the model-free approaches.

### Safe Model-free RL

- Goal:  $max_{\theta} J(\theta)$ s.t. $g(\theta) \geq 0$ by tracking performance
- Safety: $g(\theta) \geq 0, \forall t$  with probability $\geq 1 - \delta$

As we have looked, the $g(\theta)$ evaluates the trajectories and $J(\theta)$ represents the performance. Given them, we have two framework to hold the safety in RL. One is the tracking performance $max_{\theta} J(\theta)$, which contains few noisy experiments. And the other is the safety constraint $g(\theta) \geq 0$, which holds the safety for all experiments with probability $\geq 1 - \delta$

#### Bayesian Optimisation

check my reviewing note on github

https://github.com/Rowing0914/Reinforcement_Learning/blob/master/safe_reinforcement_learning/Bayes_optimisation/BayesOptimisation.pdf

#### SafeOPT

**Theorem (informal):**

Under suitable conditions on the kernel and on $J, g$ there exists a function $T(\epsilon, \delta)$ such that for any $\epsilon > 0$ and $\delta > 0$, it holds with probability at least $1 - \delta$ that 

- SafeOPT never makes an unsafe decision
- After at most $T(\epsilon, \delta)$ iterations, it found an $\epsilon$-optimal reachable point

$T(\epsilon, \delta) \in O\Big( \big( ||J||_k + ||g||_k \big)  \frac{\log^3{1/\delta}}{\epsilon^2} \Big) $

- Safe Exploration for Optimization with Gaussian Processes by Y. Sui, A. Gotovos, J.W. Burdick, A. Krause
- Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics by F.Berkenkamp, A.P. Schoellig, A. Krause
- Safe Exploration for Active Learning with Gaussian Processes by J. Schreiter, D. Nguyen-Tuong, M. Eberts, B. Bischoff, H. Markert, M. Toussaint

#### Modelling Context

- Concept is derived from CGP-UCB introduced by [Contextual Gaussian Process Bandit Optimization by Andreas Krause, Cheng Soon Ong](http://www.ong-home.my/papers/krause11cgp-ucb.pdf)

#### Case of Multiple sources of information for Bayes Optimisation

- Virtual vs. Real: Trading Off Simulations and Physical Experiments in Reinforcement Learning with Bayesian Optimization by A. Marco, F. Berkenkamp, P. Hennig, A. Schöllig, A. Krause, S. Schaal, S. Trimpe, ICRA'17

### Model-based RL

#### Rendering exploration safe

- Safe Control under Uncertainty D. Sadigh, A. Kapoor, RSS, 2016
- Safe Exploration in Markov Decision Processes T.M. Moldovan, P. Abbeel, ICML, 2012
- Safe Exploration in Finite Markov Decision Processes with Gaussian Processes M. Turchetta, F. Berkenkamp, A. Krause, NIPS, 2016
- Safe Exploration and Optimization of Constrained MDPs using Gaussian Processes Akifumi Wachi, Yanan Sui, Yisong Yue, Masahiro Ono, AAAI, 2018

#### A Bayesian Dynamics Model

Modelling the dynamics of the environment by bayes theorem.Safe and Robust Learning Control with Gaussian Processes

- In Linear case(linearise the nonlinear modelled dynamics of the environment):
  - Safe and Robust Learning Control with Gaussian Processes F. Berkenkamp, A.P. Schoellig, ECC, 2015
  - Regret Bounds for Robust Adaptive Control of the Linear Quadratic Regulator S. Dean, H. Mania, N. Matni, B. Recht, S. Tu, arXiv, 2018
- Forwards-propagating uncertain, nonlinear dynamics(Outer approximation contains true dynamics for all time steps with probability at least)
  - Learning-based Model Predictive Control for Safe Exploration T. Koller, F. Berkenkamp, M. Turchetta, A. Krause, CDC, 2018
- Model predictive control references
  - Learning-based Model Predictive Control for Safe Exploration T. Koller, F. Berkenkamp, M. Turchetta, A. Krause, CDC, 2018
  - Reachability-Based Safe Learning with Gaussian Processes A.K. Akametalu, J.F. Fisac, J.H. Gillula, S. Kaynama, M.N. Zeilinger, C.J. Tomlin, CDC, 2014
  - Robust constrained learning-based NMPC enabling reliable mobile robot path tracking C.J. Ostafew, A.P. Schoellig, T.D. Barfoot, IJRR, 2016
  - Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control S. Kamthe, M.P. Deisenroth, AISTATS, 2018
  - Chance Constrained Model Predictive Control A.T. Schwarm, M. Nikolaou, AlChE, 1999

#### Region of attraction

- Safe Model-based Reinforcement Learning with Stability Guarantees F. Berkenkamp, M. Turchetta, A.P. Schoellig, A. Krause, NIPS, 2017

#### Lyapunov function

- Safe Model-based Reinforcement Learning with Stability Guarantees F. Berkenkamp, M. Turchetta, A.P. Schoellig, A. Krause, NIPS, 2017
- Lyapunov Design for Safe Reinforcement Learning T.J. Perkings, A.G. Barto, JMLR, 2002
- The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamic Systems S.M. Richards, F. Berkenkamp, A. Krause



## Summary of reference(* mostly borrowed the words from the original paper)

#### [Safe Exploration for Optimization with Gaussian Processes by Y. Sui, A. Gotovos, J. W. Burdick, A. Krause in 2015](http://proceedings.mlr.press/v37/sui15.pdf)

- Abstract
  - We consider sequential decision problems under uncertainty, where we seek to optimise an unknown function from noisy samples. This requires balancing exploration (learning about the objective) and exploitation (localising the maximum), a problem well-studied in the multi-armed bandit literature. We tackle this novel, yet rich, set of problems under the assumption that the unknown function satisfies regularity conditions expressed via a Gaussian process prior.
- Proposal
  - SAFEOPT and its theoretical guarantee of its convergence to a natural notion of optimum reachable under safety constraints.
  - SAFEOPT: The algorithm uses Gaussian processes to make predictions about "f" based on noisy evaluations, and uses their predictive uncertainty to guide exploration.
- Experiments
  - We evaluate SAFEOPT on synthetic data, as well as two real applications:movie recommendation, and therapeutic spinal cord stimulation.
- Conclusions
  - Theoretically, we proved a bound on its sample complexity to achieve an epsilon-optimal solution, while guaranteeing safety with high probability. Experimentally, we demonstrated that SAFEOPT indeed exhibits its safety and convergence properties. We believe our results provide an important step towards employing machine learning algorithms “live” in safety-critical applications.

#### [Stagewise Safe Bayesian Optimisation with Gaussian Processes by Y. Sui, V. Zhuang, J. W. Burdick, Y. Yue in 2018](https://arxiv.org/pdf/1806.07555.pdf)

- Abstract
  - We consider the problem of optimising an unknown utility function with absolute feedback or preference feedback subject to unknown safety constraints. We develop an efficient safe Bayesian optimisation algorithm, STAGEOPT, that separates safe region expansion and utility function maximisation into two distinct stages. We provide theoretical guarantees for both the satisfaction of safety constraints as well as convergence to the optimal utility value. Their contribution is that the proposed algorithm address the challenge of efficiently identifying the total safe region and optimising the utility function within the safe region.
- Proposal
  - STAGEOPT, which is able to tackle non-comparable safety constraints and utility function
- Experiments
  - We evaluated multiple cases such as single safety function, multiple safety functions, real-valued utility, and duelling-feedback utility.
  - We evaluated our algorithm on synthetic data as well as on a live clinical experiment on spinal cord therapy.
- Conclusions
  - Our extensive experiments on synthetic data show that STAGEOPT can achieve its theoretical guarantees on safety and optimality. Its performance on safe expansion is among the best and utility maximisation outperforms the state-of-the-art.
  - Future work directions: it would be interesting to incorporate dynamics into our setting, which would lead to the multi-criteria safe reinforcement learning setting (Moldovan & Abbeel, 2012; Turchetta et al., 2016; Wachi et al., 2018) or developing theoretically rigorous approaches outside of using Gaussian processes (GPs).

#### [Safe Controller Optimization for Quadrotors with Gaussian Processes by F. Berkenkamp, A. P. Schoellig, A. Krause, ICRA 2016](http://www.dynsyslab.org/wp-content/papercite-data/pdf/berkenkamp-icra16.pdf)

* Abstract
  * Typical approach to tune the parameters used in the model is **Bayes Optimisation** though, unfortunately it does not take the safety condition into account during its exploration phase. Especially when we deal with controlling the mobile robot, we have to manage to initialise the controller with certain parameters. In this paper, we overcome this problem by applying, for the first time, a recently developed safe optimisation algorithm, SAFEOPT, to the problem of automatic controller parameter tuning. Given an initial, low-performance controller, SAFEOPT automatically optimises the parameters of a control law while guaranteeing safety. It models the underlying performance measure as a Gaussian process and only explores new controller parameters whose performance lies above a safe performance threshold with high probability
* Proposal
  *  In this paper, we overcome this problem by applying, for the first time, a recently developed safe optimisation algorithm, SAFEOPT, to the problem of automatic controller parameter tuning. And they have modified it to work without the specification of a Lipschitz constant.
* Experiments
  * we demonstrate the algorithm on a quad-rotor vehicle, a Parrot AR.Drone 2.0. A video of the experiments can be found at http://tiny.cc/icra16_video. 
* Conclusions
  * It was shown that the algorithm enables efficient, automatic, and global optimisation of the controller parameters without risking dangerous and expensive system failures.

#### [Monitoring temporal properties of continuous signals by O. Maler, D. Nickovic, FT, 2004](http://www-tpts02.imag.fr/~maler/Papers/monitor.pdf)

- Abstract
  - In this paper we introduce a variant of temporal logic(Temporal logic is a rigorous formalism for specifying desired behaviours of discrete systems such as programs or digital circuits.) tailored for specifying desired properties of continuous signals. The logic is based on a bounded subset of the real-time logic MITL, augmented with a static mapping from continuous domains into propositions. This work is, to the best of our knowledge, the first application of temporal logic monitoring to continuous and hybrid systems and we hope it will help in promoting formal methods beyond their traditional application domains.
- Proposal
  - From formulae in this logic we create automatically property monitors that can check whether a given signal of bounded length and finite variability satisfies the property
- Experiments
  - they demonstrate the behaviour of a prototype implementation of our tool on signals generated using Matlab/Simulink.
- Conclusions
  - ....



#### [Safe Control under Uncertainty by D. Sadigh, A. Kapoor, RSS, 2016](https://people.eecs.berkeley.edu/~dsadigh/Papers/sadigh-uncertainty-rss2016.pdf)

- Prerequisite: Temporal Logic (**temporal logic** is any system of rules and symbolism for representing, and reasoning about, propositions qualified in terms of [time](https://en.wikipedia.org/wiki/Time) (for example, "I am *always* hungry", "I will *eventually* be hungry", or "I will be hungry *until* I eat something"). It is sometimes also used to refer to **tense logic**, a [modal logic](https://en.wikipedia.org/wiki/Modal_logic)-based system of temporal logic introduced by [Arthur Prior](https://en.wikipedia.org/wiki/Arthur_Prior) in the late 1950s, with important contributions by [Hans Kamp](https://en.wikipedia.org/wiki/Hans_Kamp). It has been further developed by [computer scientists](https://en.wikipedia.org/wiki/Computer_scientists), notably [Amir Pnueli](https://en.wikipedia.org/wiki/Amir_Pnueli), and [logicians](https://en.wikipedia.org/wiki/Logician). [wikipedia](https://en.wikipedia.org/wiki/Temporal_logic)) or you can refer to the [course material of UC Berkeley](https://people.eecs.berkeley.edu/~sseshia/fmee/lectures/EECS294-98_Spring2014_STL_Lecture.pdf)

- Abstract
  - Safe control of dynamical systems that satisfy temporal invariants expressing various safety properties is a challenging problem. Indeed, a robotic system might employ a camera sensor and a machine learned system to identify obstacles. Consequently, the safety properties the controller has to satisfy, will be a function of the sensor data and the associated classifier. They are inspired by the concept of the relatively new Probabilistic Signal Temporal Logic (PrSTL), an expressive language to define stochastic properties, and enforce probabilistic guarantees on them. Then they have proposed an efficient algorithm to reason about safe controllers given the constraints derived from the PrSTL specification. 
- Proposal
  - They have combined conventional STL with stochastic Gaussian Process to dynamically adapt the hard rules defined by STL to the real world environment, such as autonomous driving.
  - Source code: [Controller Synthesis for Probabilistic Signal Temporal Logic Specifications](https://github.com/dsadigh/CrSPrSTL)
- Experiments
  - We demonstrate our approach by deriving safe control of quadrotors and autonomous vehicles in dynamic environments.
- Conclusions
  - The key contributions include defining PrSTL, a logic for expressing probabilistic properties that can embed Bayesian graphical models. the resulting logic adapts as more data is observed with the evolution of the system.

#### [Constrained Markov decision processes Eitan Altman, CRC Press, 1999](http://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)

- Abstract
  - please refer to the original paper..

#### [Essentials of robust control Kemin Zhou, John C. Doyle, PH, 1998](http://dl.offdownload.ir/ali/Essentials%20of%20Robust%20Control.pdf)

- Abstract
  - please refer to the original paper..

#### [Robust control of Markov decision processes with uncertain transition matrices by Arnab Nilim, Laurent El Ghaoui, OR, 2005](https://people.eecs.berkeley.edu/~elghaoui/pdffiles/rmdp_erl.pdf)

- Abstract
  - please refer to the original paper..

#### [Search-based Structured Prediction by Hal Daume III, John Langford, Daniel Marcu, ML, 2009](https://arxiv.org/pdf/0907.0786.pdf)

- Abstract
  - **SEARN** is a meta-algorithm that transforms these complex problems into simple classification problems to which any binary classifier may be applied. It is able to learn prediction functions for any loss function and any class of features. Moreover, **SEARN** comes with a strong, natural theoretical guarantee: good performance on the derived classification problems implies good performance on the structured prediction problem.
- Proposal: **SEARN**
- Experiments: NLP dataset
- Conclusions: check abstract

#### [Efficient Reductions for Imitation Learning by Stephan Ross, Drew Bagnell, AISTATS 2010](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats10-paper.pdf)

- Abstract

  - In imitation learning, the IID assumption does not hold anymore because the learned policy from the teacher is not from the same data distribution. We show that this leads to compounding errors and a regret bound that grows quadratically in the time horizon of the task. We propose two alternative algorithms(**forward training algorithm and stochastic mixing iterative learning algorithm**) for imitation learning where training occurs over several episodes of interaction. These two approaches share in common that the learner’s policy is slowly modified from executing the expert’s policy to the learned policy. We show that this leads to stronger performance guarantees and demonstrate the improved performance on two challenging problems: training a learner to play 

    - **1**) a 3D racing game (Super Tux Kart) and
    - **2**) Mario Bros.

    ; given input images from the games and corresponding actions taken by a human expert and near-optimal planner respectively.

- Proposal

  - Forward training algorithm
  - SMILe(Stochastic mixing iterative learning algorithm)

- Experiments

  - Mario cart racing game
  - Mario Bros

- Conclusions

  - We showed that SMILe works better in practice than the traditional approach on two challenging tasks. 

#### [Query-Efficient Imitation Learning for End-to-End Autonomous Driving by Jiakai Zhang, Kyunghyun Cho, AAAI, 2017](https://arxiv.org/pdf/1605.06450.pdf)

- Abstract
  - One way to approach end-to-end autonomous driving is to learn a policy function that maps from a sensory input, such as an image frame from a front-facing camera, to a driving action, by imitating an expert driver, or a reference policy. However, A policy function trained in this way however is known to suffer from unexpected behaviours due to the mismatch between the states reachable by the reference policy and trained policy functions. In this paper, we propose an extension of the DAgger, called SafeDAgger, that is query-efficient and more suitable for end-to-end autonomous driving.
- Proposal
  - In this paper, we propose an extension of the DAgger, called SafeDAgger, that is query-efficient and more suitable for end-to-end autonomous driving. And we first introduced a safety policy which prevents a primary policy from falling into a dangerous state by automatically switching between a reference policy and the primary policy without querying the reference policy
- Experiments
  - We evaluate the proposed SafeDAgger in a car racing simulator and show that it indeed requires less queries to a reference policy. 
- Conclusions
  - We observe a significant speed up in convergence, which we conjecture to be due to the effect of automated curriculum learning. The extensive experiments on simulated autonomous driving showed that the SafeDAgger not only queries a reference policy less but also trains a primary policy more efficiently

#### [EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning by Kunal Menda, Katherine Driggs-Campbell, Mykel J. Kochenderfer, arXiv2018](https://arxiv.org/pdf/1807.08364.pdf)

- Abstract
  - While imitation learning is often used in robotics, this approach often suffers from data mismatch and compounding errors. DAgger is an iterative algorithm that addresses these issues by aggregating training data from both the expert and novice policies, but does not consider the impact of safety. We present a probabilistic extension to DAgger, which attempts to quantify the confidence of the novice policy as a proxy for safety. Our method, EnsembleDAgger, approximates a GP using an ensemble of neural networks. Using the variance as a measure of confidence, we compute a decision rule that captures how much we doubt the **novice** (the opposite position of the experts), thus determining when it is safe to allow the novice to act. 
- Proposal
  - EnsembleDAgger, approximates a GP using an ensemble of neural networks. Using the variance as a measure of confidence, we compute a decision rule that captures how much we doubt the **novice** (the opposite position of the experts), thus determining when it is safe to allow the novice to act. And we aim to maximise the novice’s share of actions, while constraining the probability of failure.
- Experiments
  - they examined their approach on an inverted pendulum and in the MuJoCo HalfCheetah environment.
- Conclusions
  - To avoid requiring precise knowledge of safety, we assume the risk of a state to be inversely related to the size of the perturbation to an expert’s action that it can accept without compromising safety.
    - discrepancy rule: weighted coin-flip that decides whether the novice acts in the VANILLA-DAGGER decision rule.
    - doubt rule: the novice proposes an action that is bounded in its deviation from the expert’s choice of action, as proposed by SAFE-DAGGER* [11], but also must exhibit low variance in its choice.
  - We found that the doubt rule effectively constrains the novice to act only in states it is familiar with, i.e. states that are within some neighbourhood of states labelled in D, while the discrepancy rule haphazardly allows the novice to act in states where there is chance agreement between their actions. Though the doubt rule alone
    is shown to be superior to the discrepancy rule alone, there exist hyperaparameter settings in which the conjunction of the rules is better than either individually.

#### [Provably safe and robust learning-based model predictive control by A. Aswani, H. Gonzalez, S.S. Satry, C.Tomlin, Automatica, 2013](https://arxiv.org/pdf/1107.2487.pdf)

- Abstract
  - Controller design faces a trade-off between robustness and performance. This paper describes a learning-based model predictive control (LBMPC) scheme that provides deterministic guarantees on robustness, while statistical identification tools are used to identify richer models of the system in order to improve performance. the benefits of this framework are that it handles state and input constraints, optimises system performance with respect to a cost function, and can be designed to use a wide variety of parametric or non-parametric statistical tools. The main insight of LBMPC is that safety and performance can be decoupled under reasonable conditions in an optimisation framework by maintaining two models of the system. The first is an approximate model with bounds on its uncertainty, and the second model is updated by statistical methods. LBMPC improves performance by choosing inputs that minimise a cost subject to the learned dynamics, and it ensures safety and robustness by checking whether these same inputs keep the approximate model stable when it is subject to uncertainty. Furthermore, we show that if the system is sufficiently excited, then the LBMPC control action probabilistically converges to that of an MPC computed using the true dynamics.
- Proposal
  - LBMPC improves performance by choosing inputs that minimise a cost subject to the learned dynamics, and it ensures safety and robustness by checking whether these same inputs keep the approximate model stable when it is subject to uncertainty.
- Experiments
  - Energy-efficient building automation
  - High Performance Quadrotor Helicopter Flight
  - Example: Moore-Greizer Compressor Model
- Conclusions
  - LBMPC uses a linear model with bounds on its uncertainty to construct invariant sets that provide deterministic guarantees on robustness and safety. A simulation shows that LBMPC can improve
    over linear MPC, and experiments on testbeds show that such improvement translates to real systems



#### [Safe Reinforcement Learning via Shielding by M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Nickum, U. Topcu, AAAI, 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17211/16534)

- Abstract
  - We introduce a new approach to learn optimal policies while enforcing properties expressed in **temporal logic**. To this end, given the temporal logic specification that is to be obeyed by the learning system, we propose to synthesise a reactive system called a **shield** (a *correct-by-construction reactive system*). The **shield** monitors the actions from the learner and corrects them only if the chosen action causes a violation of the specification.
- Proposal
  - The method is based on shielding the decisions of the underlying learning algorithm from violating the specification. We proposed an algorithm for the automated synthesis of shields for given temporal logic specifications.
- Experiments
  - (1) a robot in a grid world, (2) a self-driving car scenario, (3) the water tank scenario, and (4) the pacman example.
- Conclusions
  - We demonstrated the use of shielded learning on several RL scenarios. In all of them, the learning performance of the shielded agents improved compared to the unshielded case. The main downside of our approach is that in order to prevent the learner from making unsafe actions, some approximate model of when which action is unsafe needs to be available. Our experiments show, however, that in applications in which safe learning is needed, the effort to construct an abstraction is well-spent, as our approach not only makes learning safe, but also shows great promise of improving learning performance.



#### [Linear Model Predictive Safety Certification for Learning-based Control by K.P. Wabersich, M.N. Zeilinger, CDC, 2018](https://arxiv.org/pdf/1803.08552.pdf)

- Abstract

  - While it has been repeatedly shown that learning-based controllers can provide superior performance, they often lack of safety guarantees. This paper aims at addressing this problem by introducing a model predictive safety certification (MPSC) scheme for linear systems with additive disturbances. The scheme verifies safety of a proposed learning-based input and modifies it as little as necessary in order to keep the system within a given set of constraints. By relying on robust MPC methods, the presented concept is amenable for application to large-scale systems with similar offline computational complexity as e.g. ellipsoidal safe set approximations. 

- Proposal

  - MPSC

  ![wabersich_abs](https://github.com/Rowing0914/Reinforcement_Learning/blob/master/safe_reinforcement_learning/A_comprehensive_survey_Safe_RL/images/wabersich_abs.png)

- Experiments

  - they use some artificially and uniformly sampled dataset

- Conclusions

  - By relying on robust MPC methods, the presented concept is amenable for application to large-scale systems with similar offline computational complexity as e.g. ellipsoidal safe set approximations. 



#### [Safe Exploration of State and Action Spaces in Reinforcement Learning by J. Garcia, F. Fernandez, JAIR, 2012](https://jair.org/index.php/jair/article/view/10789/25759)

- Abstract
  - While reinforcement learning is well-suited to domains with complex transition dynamics  and  high-dimensional state-action spaces, an additional challenge is posed by the need for safe and efficient exploration. Traditional exploration techniques are not particularly useful for solving dangerous tasks, where the trial and error process may lead to  the  selection  of  actions  whose  execution  in  some  states  may  result  in  damage  to  the learning system (or any other system). Consequently, when an agent begins an interaction with a dangerous and high-dimensional state-action space, an important question arises; namely, that of how to avoid (or at least minimise) damage caused by the exploration of the state-action space.  We introduce the **PI-SRL** algorithm which safely improves sub-optimal albeit robust behaviours for continuous state and action control tasks and which efficiently learns from the experience gained from the environment.  We evaluate the proposed method in  four  complex  tasks:   automatic  car  parking,  pole-balancing,  helicopter  hovering,  and business management.
- Proposal
  - **PI-SRL** algorithm which safely improves sub-optimal albeit robust behaviours for continuous state and action control tasks and which efficiently learns from the experience gained from the environment. 
- Experiments
  - We evaluate the proposed method in  four  complex  tasks:   automatic  car  parking,  pole-balancing,  helicopter  hovering,  and business management.
- Conclusions
  -  The main contributions of this algorithm are the definitions of a novel case-based risk function and a baseline behaviour for the safe exploration of  the  state-action  space.   The  use  of  the  case-based  risk  function  presented  is  possible inasmuch  as  the  policy  is  stored  as  a  case-base.



#### [Safe Exploration in Continuous Action Spaces by G. Dalai, K. Dvijotham, M. Veccerik, T. Hester, C. Paduraru, Y. Tassa, arXiv, 2018](https://arxiv.org/pdf/1801.08757.pdf)

- Abstract
  - We address the problem of deploying a reinforcement learning (RL) agent on a physical system such as a datacenter cooling unit or robot, where critical constraints must never be violated. Our
    technique is to directly add to the policy a safety layer that analytically solves an action correction formulation per each state. The novelty of obtaining an elegant closed-form solution is attained due to a linearised model, learned on past trajectories consisting of arbitrary actions This is to mimic the real-world circumstances where data logs were generated with a behaviour policy that is implausible to describe mathematically; such cases render the known safety-aware off-policy methods inapplicable. We demonstrate the efficacy of our approach on new representative physics-based environments, and prevail where reward shaping fails by maintaining zero constraint violations.

- Proposal

  - we proposed a state-based action correction mechanism, which accomplishes the goal of zero-constraint-violations in tasks where the agent is constrained to a confined region.

  ![wabersich_abs (https://github.com/Rowing0914/Reinforcement_Learning/blob/master/safe_reinforcement_learning/A_comprehensive_survey_Safe_RL/images/g_dalai18.png)

- Experiments

  - https://youtu.be/KgMvxVST-9U
  - https://www.youtube.com/watch?v=yr6y4Mb1ktI&feature=youtu.be

- Conclusions

  - The resulting gain is not only in maintaining safety but also in enhanced performance in terms of reward. This suggests our method promotes more efficient exploration – it guides the exploratory actions in the direction of feasible policies. Since our solution is stand-alone and applied directly at the policy level, it is independent of the RL algorithm used and can be plugged into any other continuous control algorithm

#### [High Confidence Policy Improvement by Philip S. Thomas, Georgios Theocharous, Mohammad Havamzadeh, ICML 2015](https://people.cs.umass.edu/~pthomas/papers/Thomas2015b.pdf)

- Abstract
  - We present a batch reinforcement learning (RL) algorithm that provides probabilistic guarantees about the quality of each policy that it proposes, and which has no hyper-parameters that require expert tuning. The algorithm requires user to select two things, a lower-bound and the confidence level. Then the proposed algo will ensure that the probability that it returns never breach the condition. We then propose an incremental algorithm that executes our policy improvement algorithm repeatedly to generate multiple policy improvements
- Proposal
  - batch (semi-)safe policy improvement algorithm, **POLICYIMPROVEMENT** , takes as input a set of trajectories labelled with the policies that generated them a performance lower bound, and a confidence level, and outputs either a new policy or NO SOLUTION FOUND (NSF).
- Experiments
  - a discrete 4 × 4 grid-world, the canonical Mountain Car domain
  - The digital marketing domain involves optimising a policy that targets advertisements towards each user that visits a web-page, and uses real data collected from a Fortune 20 company. 
- Conclusions
  - We have presented batch and incremental policy improvement algorithms that provide (exact and approximate) statistical guarantees about the performance of policies that they propose. These guarantees can be tuned by the user to account for the acceptable level of risk in each application. We showed on a real world digital marketing problem that our algorithms can use a realistic amount of data to provide guaranteed policy improvements with confidence 95%.

#### [Safe and efficient off-policy reinforcement learning by Remi Munos, Thomas Stepleton, Anna Harutyunyan, Marc G. Bellemare, NIPS, 2016](https://papers.nips.cc/paper/6538-safe-and-efficient-off-policy-reinforcement-learning.pdf)

- Abstract
  - In this work, we take a fresh look at some old and new algorithms for off-policy, return-based reinforcement learning. Expressing these in a common form, we derive a novel algorithm, Retrace($\lambda$) with three desired properties:
    - it has low variance
    - it safely uses samples collected from any behaviour policy, whatever its degree of "off-policyness"
    - it is efficient as it makes the best use of samples collected from near on-policy behaviour policies

    and they have examined their proposal on a standard suite of Atari 2600 games. In addition, we provide as **a corollary the first proof of convergence of Watkins’ Q(λ)** (see, e.g., Watkins, 1989; Sutton and Barto, 1998).

- Proposal

  - Retrace(λ) uses an importance sampling ratio truncated at 1. Compared to IS, it does not suffer from the variance explosion of the product of IS ratios. Now, similarly to Qπ(λ) and unlike TB(λ), it does not cut the traces in the on-policy case, making it possible to benefit from the full returns. In the off-policy case, the traces are safely cut, similarly to TB(λ)
  - this algorithm does not require GLIE(Greedy in the Limit with Infinite Exploration, Singh et al.,
    2000) assumption.

- Experiments

  - To validate our theoretical results, we employ Retrace(λ) in an experience replay (Lin, 1993) setting, where sample transitions are stored within a large but bounded replay memory and subsequently replayed as if they were new experience. We compare our algorithms’ performance on 60 different Atari 2600 games in the Arcade Learning Environment (Bellemare et al., 2013) using Bellemare et al.’s inter-algorithm score distribution. Inter-algorithm scores are normalized so that 0 and 1 respectively correspond to the worst and best score for a particular game, within the set of algorithms under comparison.

- Conclusions

  - Retrace(λ) can be seen as an algorithm that automatically adjusts – efficiently and safely – the length of the return to the degree of ”off-policyness” of any available data.

#### [Constrained Policy Optimisation by Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel, ICML, 2017](https://arxiv.org/pdf/1705.10528.pdf)

- Abstract
  - For many applications of reinforcement learning it can be more convenient to specify both a reward function and constraints, rather than trying to design behaviour through the reward function.   We propose Constrained Policy Optimisation (CPO), the first general-purpose policy search algorithm for constrained reinforcement learning with guarantees for near-constraint satisfaction at each iteration. Our method allows us to train neural network policies for high-dimensional control while making guarantees about policy behaviour all throughout training. Our guarantees are based on a new theoretical result, which is of independent interest: we prove a bound relating the expected returns of two policies to an average divergence between them. We demonstrate the effectiveness of our approach on simulated robot locomotion tasks where the agent must satisfy constraints motivated by safety
- Proposal
  - Constrained Policy Optimisation (CPO), the first general-purpose policy search algorithm for constrained reinforcement learning with guarantees for near-constraint satisfaction at each iteration.
- Experiments
  - Using two tasks below, they have attempted to answer the questions below.
  - Experimental Tasks
    - **Circle**: The agent is rewarded for running in a wide-circle, but is constrained to stay within a 	safe region smaller than the radius of the target circle.
    - **Gather**: The agent is rewarded for collecting green apples, and constrained to avoid red bombs.
  - Questions
    - Does CPO succeed at enforcing behavioural constraints when training neural network policies with thousands of parameters?
    - How does CPO compare with a baseline that uses primal-dual optimisation? Does CPO behave better with respect to constraints?
    - How much does it help to constrain a cost upper bound, instead of directly constraining the cost?
    - What benefits are conferred by using constraints instead of fixed penalties?
- Conclusions
  - In this article, we showed that a particular optimisation problem results in policy updates that are guaranteed to both improve return and satisfy constraints. This enabled the development of CPO, our policy search algorithm for CMDPs, which approximates the theoretically-guaranteed algorithm in a principled way. We demonstrated that CPO can train neural network policies with thousands of parameters on high-dimensional constrained control tasks, simultaneously maximising reward and approximately satisfying constraints. Our work represents a step towards applying reinforcement learning in the real world, where constraints on agent behaviour are sometimes necessary for the sake of safety.

#### [Safe Exploration for Active Learning with Gaussian Processes by J. Schreiter, D. Nguyen-Tuong, M. Eberts, B. Bischoff, H. Markert, M. Toussaint](https://pdfs.semanticscholar.org/9afd/7874507eacdd0adc8e59b1e9ad5f0ce068a3.pdf)

- Abstract
  - In this paper, the problem of safe exploration in the active learning context is considered. Especially, they focus on the industrial system, e.g., combustion engines and gas turbines, where critical and unsafe measurements need to be avoided. The objective is to learn data-based regression models from such technical systems using a limited budget of measured, i.e. labelled, points while ensuring that critical regions of the considered systems are avoided during measurements. We propose an approach for learning such models and exploring new data regions based on Gaussian processes (GP’s). In particular, we employ a problem specific GP classifier to identify safe and unsafe regions, while using a differential entropy criterion for exploring relevant data regions. A theoretical analysis is shown for the proposed algorithm, where we provide an upper bound for the probability of failure.
- Proposal
  - Safe Active Learning with GPs
- Experiments
  - To demonstrate the efficiency and robustness of our safe exploration scheme in the active learning setting, we test the approach on a policy exploration task for the inverse pendulum hold up problem.
- Conclusions
  - Empirical evaluations on a toy example and on a policy search task for the inverse pendulum control confirm the safety bounds provided by the approach. Moreover, the experiments show the effectiveness of the presented algorithm, when selecting a near optimal input design, even under the induced safety constraint. The next steps will include evaluations on physical systems and – on the theoretical side – the error bounding of the resulting regression model, which is generally a hard problem.

#### [Bayesian Optimization with Safety Constraints: Safe and Automatic Parameter Tuning in Robotics by F.Berkenkamp, A.P. Schoellig, A. Krause](https://arxiv.org/pdf/1602.04450.pdf)

- Abstract
  - In robotics, Manual parameter tuning has been dominant in practice. Optimisation algorithms, such as Bayesian optimisation, have been used to automate this process. However, these methods may evaluate unsafe parameters during the optimisation process that lead to safety-critical system failures. SAFEOPT was developed recently and it guarantees that the performance of the system never falls below a critical value. However, coupling performance and safety is often not desirable in robotics. For example, high-gain controllers might achieve low average tracking error (performance), but can overshoot and violate input constraints. In this paper, we present a generalised algorithm that allows for multiple safety constraints separate from the objective. Given an initial set of safe parameters, the algorithm maximises performance but only evaluates parameters that satisfy safety for all constraints with high probability.
- Proposal
  - SAFEOPT-MC (Multiple Constraints): the extension of SAFEOPT. Concept is derived from CGP-UCB introduced by [Contextual Gaussian Process Bandit Optimization by Andreas Krause, Cheng Soon Ong
  - 1. Expanding the region of the optimization problem that is known to be feasible or safe as much as possible without violating the constraints,
    2. Finding the optimal parameters within the current safe set.
- Experiments
  - we demonstrate Algorithm in experiments on a quadrotor vehicle, a Parrot AR.Drone 2.0.
  - they chose Matern kernel for the kernel function in GP
- Conclusions
  - We presented a generalization of the Safe Bayesian Optimization algorithm of Sui et al. (2015) that allows multiple, separate safety constraints to be specified and applied it to nonlinear control problems on a quadrotor vehicle.

#### [Virtual vs. Real: Trading Off Simulations and Physical Experiments in Reinforcement Learning with Bayesian Optimization by A. Marco, F. Berkenkamp, P. Hennig, A. Schöllig, A. Krause, S. Schaal, S. Trimpe, ICRA'17](https://las.inf.ethz.ch/files/ewrl18_SafeRL_tutorial.pdf)

- Abstract
  - In practice, the parameters of control policies are often tuned manually. RL seems effective for this problem though, it requires too many experiments to be practical. In this paper, we propose a solution to this problem by exploiting prior knowledge from simulations, which are readily available for most robotic platforms. Specifically, we extend Entropy Search, a Bayesian optimization algorithm that maximizes information gain from each experiment, to the case of multiple information sources. The result is a principled way to automatically combine cheap, but inaccurate information from simulations with expensive and accurate physical experiments in a cost-effective manner.
- Proposal
  - In this paper, we present a Bayesian optimization algorithm for multiple information sources. We use entropy to measure the information content of simulations and experiments. Since this is an appropriate unit of measure for the utility of both sources, our algorithm is able to compare physically meaningful quantities in the same units on either side, and trade off accuracy for cost.
- Experiments
  - We apply the resulting method to a cart-pole system, which confirms that the algorithm can find good control policies with fewer experiments than standard Bayesian optimization on the physical system only.
- Conclusions
  - The main contributions of the paper are (i) a novel Bayesian optimisation algorithm that can trade off between costs of multiple information sources and (ii) the first application of such a framework to the problem of reinforcement learning and optimisation of controller parameters

#### [Safe Exploration in Markov Decision Processes T.M. Moldovan, P. Abbeel, ICML, 2012](https://icml.cc/2012/papers/838.pdf)

- Abstract
  - In environments with uncertain dynamics exploration is necessary to learn how to perform well. Existing reinforcement learning algorithms provide strong exploration guarantees, but they tend to rely on an **ergodicity assumption**. The essence of ergodicity is that any state is eventually reachable from any other state by following a suitable policy. This assumption allows for exploration algorithms that operate by simply favouring states that have rarely been visited before. But in practice, most physical systems don’t satisfy the ergodicity assumption. In this paper we address the need for safe exploration methods in Markov decision processes. We first propose a general formulation of safety through ergodicity. We then present an efficient algorithm for guaranteed safe, but potentially sub-optimal, exploration. At the core is an optimisation formulation in which the constraints restrict attention to a subset of the guaranteed safe policies and the objective flavors exploration policies.
- Proposal
  - they have proposed the Safe exploration algorithm which works on the ergodicity environment by finding the safe and potentially sub-optimal policies.
- Experiments
  - Our experiments, which include a Martian terrain exploration problem, show that our method is able to explore better than classical exploration methods.
- Conclusions
  - In addition to the safety formulation, out framework also supports a number of other safety criteria such that, Stricter ergodicity ensuring that return is possible within some horizon, H, not just eventually, with probability and so on.

#### [Safe Exploration in Finite Markov Decision Processes with Gaussian Processes M. Turchetta, F. Berkenkamp, A. Krause, NIPS, 2016](https://arxiv.org/pdf/1606.04753.pdf)

- Abstract

  - In this paper, we address the problem of safely exploring finite Markov decision processes (MDP). We define safety in terms of an a priori unknown safety constraint that depends on states and actions and satisfies certain regularity conditions expressed via a Gaussian process prior. We develop a novel algorithm, SAFEMDP, for this task and prove that it completely explores the safely reachable part of the MDP without violating the safety constraint. To achieve this, it cautiously explores safe states and actions in order to gain statistical confidence about the safety of unvisited state-action pairs from noisy observations collected while navigating the environment.
- Proposal

  - SAFEMDP
  - We introduce SAFEMDP, a novel algorithm for safe exploration in MDPs. We model safety via an a priori unknown constraint that depends on state-action pairs. Starting from an initial set of states and actions that are known to satisfy the safety constraint, the algorithm exploits the regularity assumptions on the constraint function in order to determine if nearby, unvisited states are safe. This leads to safe exploration, where only state-actions pairs that are known to fulfil the safety constraint are evaluated
- Experiments

  - We demonstrate our method on digital terrain models for the task of exploring an unknown map with a rover
- Conclusions
  - We presented SAFEMDP, an algorithm to safely explore a priori unknown environments. We used a Gaussian process to model the safety constraints, which allows the algorithm to reason about the safety of state-action pairs before visiting them. An important aspect of the algorithm is that it considers the transition dynamics of the MDP in order to ensure that there is a safe return route before visiting states. We proved that the algorithm is capable of exploring the full safely reachable region with few measurements, and demonstrated its practicality and performance in experiments.

#### [Safe Exploration and Optimization of Constrained MDPs using Gaussian Processes Akifumi Wachi, Yanan Sui, Yisong Yue, Masahiro Ono, AAAI, 2018](http://www.yisongyue.com/publications/aaai2018_safe_mdp.pdf)

- Abstract
  - Extension of SAFEMDP
  - We propose a novel approach to balance this trade-off(exploring the safety function, exploring the reward function, and exploiting acquired knowledge to maximise reward.). Specifically, our approach explores unvisited states *selectively*; that is, it priorities the exploration of a state if visiting that state significantly improves the knowledge on the achievable cumulative reward. Our approach relies on a novel information gain criterion based on Gaussian Process representations of the reward and safety functions.
- Proposal
  - their approach can account for the slippage of the mars-rover
  - their r approach employs two MDPs, which we call Optimistic and Pessimistic MDPs, and uses the difference in the value functions as the information gain criterion. Our GP safety function yields three classes of states: safe, unsafe, and uncertain. The only difference between the Optimistic and Pessimistic MDPs is that uncertain states are considered safe former and unsafe in the latter. Using this criterion, the agent is motivated to explore uncertain states that could result in high cumulative reward if they are determined safe
- Experiments
  - We demonstrate the effectiveness of our approach on a range of experiments, including a simulation using the real Martian terrain data.
- Conclusions
  - We presented a novel approach for exploring and optimising safety constrained MDPs. By modelling a priori unknown reward and safety via GPs, an agent can classify state space into safe, uncertain, and unsafe regions.

#### [Safe and Robust Learning Control with Gaussian Processes F. Berkenkamp, A.P. Schoellig, ECC, 2015](http://www.dynsyslab.org/wp-content/papercite-data/pdf/berkenkamp-ecc15.pdf)

- Abstract
  - This paper introduces a learning-based robust control algorithm that provides robust stability and performance guarantees during learning. Traditional robust control approaches have not considered online adaptation of the model and its uncertainty before. As a result, their controllers do not improve performance during operation. The approach uses Gaussian process (GP) regression based on data gathered during operation to update an initial model of the system and to gradually decrease the uncertainty related to this model. Embedding this data-based update scheme in a robust control framework guarantees stability during the learning process. In particular, this paper considers a stabilization task, linearizes the nonlinear, GP-based model around a desired operating point, and solves a convex optimization problem to obtain a linear robust controller.
- Proposal
  - In this paper, a method that combines online learning with robust control theory has been introduced with the goal of designing a learning controller that guarantees stability while gradually improving performance.
- Experiments
  - The resulting performance improvements due to the learning-based controller are demonstrated in experiments on a quadrotor vehicle
- Conclusions
  - In this paper, a method that combines online learning with robust control theory has been introduced with the goal of designing a learning controller that guarantees stability while gradually improving performance. Experiments on a quadrotor vehicle showed that the controller performance improved as more data became available. Ultimately, the GP framework has proven to be a powerful tool to combine nonlinear learning methods with standard robust control theory.

#### [Learning-based Model Predictive Control for Safe Exploration T. Koller, F. Berkenkamp, M. Turchetta, A. Krause, CDC, 2018](https://arxiv.org/pdf/1803.08287.pdf)

- Abstract
  - In model-based reinforcement learning, we aim to learn the dynamics of an unknown system from data, and based on the model, derive a policy that optimises the long-term behaviour of the system. However, these methods typically do not provide any safety guarantees, which prevents their use in safety-critical, real-world applications. In this paper, we present a learning-based model predictive control scheme that can provide provable high-probability safety guarantees. Unlike
    previous approaches, we do not assume that model uncertainties are independent. Based on these predictions, we guarantee that trajectories satisfy safety constraints. Moreover, we use a terminal set constraint to recursively guarantee the existence of safe control actions at every iteration.
- Proposal
  - SafeMPC algorithm
  - We combine ideas from robust control and GP-based RL to design a MPC(MODEL PREDICTIVE CONTROL) scheme that recursively guarantees the existence of a safety trajectory that satisfies the constraints of the system
- Experiments
  - We apply the algorithm to safely explore the dynamics of an inverted pendulum simulation.
- Conclusions
  - We introduced SAFEMPC, a learning-based MPC scheme that can safely explore partially unknown systems. The algorithm is based on a novel uncertainty propagation technique that uses a reliable statistical model of the system. As we gather more data from the system and update our statistical mode, the model becomes more accurate and control performance improves, all while maintaining safety guarantees throughout the learning process

#### [Reachability-Based Safe Learning with Gaussian Processes A.K. Akametalu, J.F. Fisac, J.H. Gillula, S. Kaynama, M.N. Zeilinger, C.J. Tomlin, CDC, 2014](http://www.ece.ubc.ca/~kaynama/papers/CDC2014_safelearning.pdf)

- Abstract
  - Recent approaches successfully introduce safety based on reachability analysis, determining a safe region of the state space where the system can operate. However, overly constraining the freedom of the system can negatively affect performance, while attempting to learn less conservative safety constraints might fail to preserve safety if the learned constraints are inaccurate. We propose a novel method that uses a principled approach to learn the system’s unknown dynamics based on a Gaussian process model and iteratively approximates the maximal safe set. 
- Proposal
  - GP regression is used to infer the disturbance set from past observations of the dynamics; this disturbance set is used to conduct reachability analysis and obtain a safety function and an optimal safe control policy.
- Experiments
  - We demonstrate our algorithm on simulations of a cart-pole system and on an experimental quadrotor application and show how our proposed scheme succeeds in preserving safety where current approaches fail to avoid an unsafe condition.
- Conclusions
  - We have introduced a general reachability-based safe learning algorithm that leverages GPs to learn a model of the system disturbances and employs a novel control strategy based on online model validation, providing stronger safety guarantees than current state-of-the-art reachability-based frameworks. 

#### [Robust constrained learning-based NMPC enabling reliable mobile robot path tracking C.J. Ostafew, A.P. Schoellig, T.D. Barfoot, IJRR, 2016](http://www.dynsyslab.org/wp-content/papercite-data/pdf/ostafew-ijrr16.pdf)

- Abstract
  - This paper presents a Robust Constrained Learning-based Nonlinear Model Predictive Control (RC LB-NMPC) algorithm for path-tracking in off-road terrain. In this work our goal is to use learning to generate low-uncertainty, non-parametric models in situ. Based on these models, the predictive controller computes both linear and angular velocities in real-time, such that the robot drives at or near its capabilities while respecting path and localisation constraints. The result is a robust, learning controller that provides safe, conservative control during initial trials when model uncertainty is high and converges to high-performance, optimal control during later trials when model uncertainty is reduced with experience
- Proposal
  - In previous work, we demonstrated unconstrained LB-NMPC, where tracking errors were reduced using real-world experience instead of pre-programming accurate analytical models (Ostafew et al., 2015a). This work represents a major extension where we use the learned model uncertainty to compute and apply robust state and input constraints.
  - This paper presents a Robust Constrained Learning-based Nonlinear Model Predictive Control (RC LB-NMPC) algorithm for path-tracking in off-road terrain.
- Experiments
  - The paper presents experimental results, including over 5 km of travel by a 900 kg skid-steered robot at speeds of up to 2.0 m/s. 
- Conclusions
  - In summary, this paper presents a Robust Constrained Learning-based Nonlinear Model Predictive Control (RCLB-NMPC) algorithm for a path-repeating, mobile robot operating in challenging off-road terrain. The goal is to guarantee constraint satisfaction while increasing performance through learning.

#### [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control S. Kamthe, M.P. Deisenroth, AISTATS, 2018](http://proceedings.mlr.press/v84/kamthe18a/kamthe18a.pdf)

- Abstract
  - To reduce the number of system interactions while simultaneously handling constraints, we propose a modelbased RL framework based on probabilistic Model Predictive Control (MPC). In particular, we propose to learn a probabilistic transition model using Gaussian Processes (GPs) to incorporate model uncertainty into longterm predictions, thereby, reducing the impact of model errors. We then use MPC to find a control sequence that minimises the expected long-term cost.
- Proposal
  - The contributions of this paper are the following: 1) We propose a new ‘deterministic’ formulation for probabilistic MPC with learned GP models and uncertainty propagation for long-term planning. 2) This reformulation allows us to apply [Pontryagin’s Maximum Principle (PMP)](https://en.wikipedia.org/wiki/Pontryagin%27s_maximum_principle) for the open-loop planning stage of probabilistic MPC with GPs. Using the PMP we can handle control constraints in a principled fashion while still maintaining necessary conditions for optimality. 3) The proposed algorithm is not only theoretically justified by optimal control theory, but also achieves a state-of-the-art data efficiency in RL while maintaining the probabilistic formulation. 4) Our method can handle state and control constraints while preserving its data efficiency and optimality properties.
- Experiments
  - We evaluate the quality of our algorithm in two ways: First, we assess whether probabilistic MPC leads to faster learning compared with PILCO, the current state of the art in terms of data efficiency. Second, we assess the impact of state constraints while performing the same task.
- Conclusions
  - We proposed an algorithm for data-efficient RL that is based on probabilistic MPC with learned transition models using Gaussian processes. By exploiting Pontryagin’s maximum principle our algorithm can naturally deal with state and control constraints.

#### [Safe Model-based Reinforcement Learning with Stability Guarantees F. Berkenkamp, M. Turchetta, A.P. Schoellig, A. Krause, NIPS, 2017](https://papers.nips.cc/paper/6692-safe-model-based-reinforcement-learning-with-stability-guarantees.pdf)

- Abstract
  - Reinforcement learning is a powerful paradigm for learning optimal policies from experimental data. However, to find optimal policies, most reinforcement learning algorithms explore all possible actions, which may be harmful for real-world systems. In this paper, we present a learning algorithm that explicitly considers safety, defined in terms of stability guarantees. Specifically, we extend control-theoretic results on Lyapunov stability verification and show how to use statistical models of the dynamics to obtain high-performance control policies with provable stability certificates.
- Proposal
  - SAFELYAPUNOVLEARNING
  - To satisfy the specified safety constraints for safe learning, we require a tool
    to determine whether individual states and actions are safe. In control theory, this safety is defined
    through the region of attraction, which can be computed for a fixed policy using Lyapunov functions.
  - We introduce a novel algorithm that can safely optimize policies in continuous state-action spaces while providing high-probability safety guarantees in terms of stability
- Experiments
  - In our experiments, we show how the resulting algorithm can safely optimise a neural network policy on a simulated inverted pendulum, without the pendulum ever falling down
- Conclusions
  - we showed how to safely optimise policies and give stability certificates based on statistical models of the dynamics. Moreover, we provided theoretical safety and exploration guarantees for an algorithm that can drive the system to desired state-action pairs during learning

#### [Lyapunov Design for Safe Reinforcement Learning T.J. Perkings, A.G. Barto, JMLR, 2002](http://www.jmlr.org/papers/volume3/perkins02a/perkins02a.pdf)