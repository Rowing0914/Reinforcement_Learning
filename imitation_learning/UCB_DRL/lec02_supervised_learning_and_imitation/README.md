## URL

http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-2.pdf



## Notes ~Supervised Learning of Behaviours~ 

### Definition of sequential decision problems


$$
s_t: \text{state} \\
o_t: \text{observation} \\
a_t: \text{action} \\
\pi_{\theta}(a_t | o_t): \text{policy} \\
\pi_{\theta}(a_t | s_t): \text{policy}
$$

### Imitation learning: supervised learning for decision making

![imitation_learning](/home/noio0925/Desktop/research/Reinforcement_Learning/imitation_learning/UCB_DRL/lec02_supervised_learning_and_imitation/images/slide_7.png)



#### Does direct imitation work?

##### Drift of distribution

Simple behaviour cloning does not work often. Because of drifting the distribution problem. Once the agent takes a different action than the one of a human driver, it will encounter a new state where the agent would not be able to perform well.

![](/home/noio0925/Desktop/research/Reinforcement_Learning/imitation_learning/UCB_DRL/lec02_supervised_learning_and_imitation/images/slide13.png)

#### How can we make it work more often?

##### DAgger

There some approaches to address this problem, but in this lecture they are discussing the approach called, **DAgger(Dataset Aggregation)**.

![](/home/noio0925/Desktop/research/Reinforcement_Learning/imitation_learning/UCB_DRL/lec02_supervised_learning_and_imitation/images/slide14.png)

##### What can cause the drifting the distribution problem?

1. Non-Markovian Behaviour
   - As a human, we don't react to the same scene as before even though we encountered it twice.
   - So the context is important for sequential decision making.
   - **Typically, LSTM cells work better here**
2. Multimodal Behaviour: Normally human behaviours are not easy to model since it can contain a lot of unreasonable factor in it.

### Case studies of recent work in IL

1. trail following as classification in Visual-Based Navigation: [A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots by A.Giusti et al., 2015](https://www.ifi.uzh.ch/dam/jcr:38859211-28c8-43c9-af40-cb6d5bea0bbe/RAL16_Giusti.pdf)
2. Imitation with LSTMs: [From Virtual Demonstration to Real-World Manipulation Using LSTM and MDN by R.Rahmatizadeh et al., 2016](https://arxiv.org/pdf/1603.03833.pdf)
3. Follow-up: adding vision: [Vision-Based Multi-Task Manipulation for Inexpensive Robots Using End-To-End Learning from Demonstration by R.Rahmatizadeh et al., 2016 et al., 2017](https://arxiv.org/pdf/1707.02920.pdf)



### What is missing from IL

#### Imitation learning: what’s the problem?

1. Humans need to provide data, which is typically finite
2. Humans are not good at providing some kinds of actions
3. Humans can learn autonomously; can our machines do the same?