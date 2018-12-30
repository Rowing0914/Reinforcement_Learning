## Profile

- Title: Tutorial on Safe Reinforcement Learning
- Authors: Felix Berkenkamp, Andreas Krause
- Organisation: ETH Zurich
- Venue: @EWRL, October 1 2018



## Notes

**Agenda**

- Specifying safety requirements and quantify risk
  - Examples(When do we need to consider the safety?)
    - Therapeutic Spinal Cord Stimulation
      - Safe Exploration for Optimization with Gaussian Processes by Y. Sui, A. Gotovos, J. W. Burdick, A. Krause
      - Stagewise Safe Bayesian Optimization with Gaussian Processes by Y. Sui, V. Zhuang, J. W. Burdick, Y. Yue
    - Safe Controller Tuning
      - Safe Controller Optimization for Quadrotors with Gaussian Processes by F. Berkenkamp, A. P. Schoellig, A. Krause, ICRA 2016
  - Safety Criterion
    - Stochastic environment/policy
    - Expected safety can be misleading
    - Expected safety and variance
    - Risk Sensitivity
- Acting safely in known environments
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

