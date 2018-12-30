## Introduction

This is the notes for the research paper below

* [A Comprehensive Survey on Safe Reinforcement Learning](http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) by Javier Garcia and Fernando Fernandez in 2015



## Notes

### Introduction: the importance of safety

in some situations in which the safety of the agent is particularly important, for example, inexpensive robotic platforms, researchers are paying increasing attention not only to the long-term reward maximisation, but also to damage avoidance (Mihatsch and Neuneier, 2002; Hans et al., 2008; Mart´ın H. and Lope, 2009; Koppejan and Whiteson, 2011; Garc´ıa and Fern´andez, 2012). [p1]



### optimality criterion-related approaches

In many works, the risk is related to the stochasticity of the environment and with the fact that, in those environments, even an optimal policy (with respect the return) may perform poorly in some cases (Coraluppi and Marcus, 1999; Heger, 1994b). [p1]

Indeed, Maximising the long-term reward does not necessarily avoid the disastrous situation. For example, the long-term reward maximisation is transformed to include some notion of risk related to the variance of the return (Howard and Matheson, 1972; Sato et al., 2002) or its worst-outcome (Heger, 1994b; Borkar, 2002; Gaskett, 2003). the optimization criterion is transformed to include the probability of visiting error states (Geibel and Wysotzki, 2005), or transforming the temporal differences to more heavily weighted events that are unexpectedly bad (Mihatsch and Neuneier, 2002). [p2]



### exprolation process

**Problem is that** most of those exploration methods are blind to the risk of actions.

To avoid risky situations, the exploration process is often modified by including prior knowledge of the task. And This prior knowledge can be used to provide initial information to the RL algorithm biasing the subsequent exploratory process (Driessens and Dˇzeroski, 2004; Mart´ın H. and Lope, 2009; Koppejan and Whiteson, 2011), to provide a finite set of demonstrations on the task (Abbeel and Ng, 2005; Abbeel et al., 2010), or to provide guidance (Clouse, 1997; Garc´ıa and Fern´andez, 2012).[p2]



initial knowledge was used to bootstrap an evolutionary approach by the winner of the helicopter control task of the 2009 RL competition (Mart´ın H. and Lope, 2009). In this approach, several neural networks that clone error-free teacher policies are added to the initial population (facilitating the rapid convergence of the algorithm to a near-optimal policy and, indirectly, reducing agent damage or injury).[p2]

**apprentice learning**

Abbeel and Ng (2005); Abbeel et al. (2010) use a finite set of demonstrations from a teacher to derive a safety policy for the helicopter control task, while minimizing the helicopter crashes. [p2]

the guidance provided by a teacher during the exploratory process has also been demonstrated to be an effective method to avoid dangerous or catastrophic states (Garc´ıa and Fern´andez, 2012).[p2]



### definition of Safe RL

Safe RL can be defined as the process of learning policies that maximize the expectation of the return in problems in which it is important to ensure reasonable system performance and/or respect safety constraints during the learning and/or deployment processes. [p3]