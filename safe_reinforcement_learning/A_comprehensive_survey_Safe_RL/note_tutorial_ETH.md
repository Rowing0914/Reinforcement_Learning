## Profile

- Title: [Tutorial on Safe Reinforcement Learning](https://las.inf.ethz.ch/files/ewrl18_SafeRL_tutorial.pdf)
- Authors: Felix Berkenkamp, Andreas Krause
- Organisation: ETH Zurich
- Venue: @EWRL, October 1 2018



## Prerequisites

### Bayesian Optimisation

- **[Introduction to Bayesian Optimization by Javier Gonzalez Masterclass, 7-February, 2107 @Lancaster University](http://gpss.cc/gpmc17/slides/LancasterMasterclass_1.pdf)**
- [**A Tutorial on Bayesian Optimization for Machine Learning by Ryan P. Adams School of Engineering and Applied Sciences Harvard University**](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf)

### Temporal Logic

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





## Notes

**Agenda**

- Specifying safety requirements and quantify risk
  - Examples(When do we need to consider the safety?) *details in Summary of reference
    - Therapeutic Spinal Cord Stimulation
      - Safe Exploration for Optimization with Gaussian Processes by Y. Sui, A. Gotovos, J. W. Burdick, A. Krause
      - Stagewise Safe Bayesian Optimization with Gaussian Processes by Y. Sui, V. Zhuang, J. W. Burdick, Y. Yue
    - Safe Controller Tuning
      - Safe Controller Optimization for Quadrotors with Gaussian Processes by F. Berkenkamp, A. P. Schoellig, A. Krause, ICRA 2016
  - Safety Criterion(have a look at **Bayesian optimisation** and **temporal logic** beforehand)
    - specifying safety behaviour(safety is the similar concept as avoiding bad trajectories)
      - $g(\{ s_t, a_t \}^N_{t=0} ) = g(\tau) > 0$ : Monitoring temporal properties of continuous signals by O. Maler, D. Nickovic, FT, 2004
      - $g(\tau) = \min_{t=1:N} \Delta(s_t, a_t)$ : Safe Control under Uncertainty by D. Sadigh, A. Kapoor, RSS, 2016
    - But the expected safety is not perfect rather misleading indication, since it averages the bad and good trajectories. So, we need to consider the range of the distribution of trajectories, e.g., the variance.
    - Notion of safety
      - Expected risk: $E[G]$
      - Moment penalised: $E[e^{\tau G}]$
      - Value at Risk: $VaR_{\delta}[G] = \inf \{ \epsilon \in R: p(G \leq \epsilon) \} \geq \delta $
      - Conditional Value at Risk: $CVaR_{\delta}[G] = \frac{1}{\delta} \int^{\delta}_0 VaR_{\alpha}[G] d\alpha$
      - Worst-case: $g(\tau) > 0 \ \forall_{\tau} \in \Gamma$

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

      - A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell, 2011

  - Acting safely in unknown environments

- Safe exploration (model-free and model-based)



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