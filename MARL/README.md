## Outline

1. Introduction
2. Background: RL in Single/Multi agent case
   1. Single Agent
   2. Multi Agent
   3. Static, repeated and stage games
3. Benefits and Challenges
   1. Benefits
   2. Challenges
4. MARL goal
5. Taxonomy of MARL algorithms
6. MARL algorithms
   1. Fully Cooperative tasks
      1. Coordination-Free methods
      2. Coordination-Based methods
      3. Indirect Coordination methods
   2. Fully Competitive tasks
   3. Mixed Tasks
      1. Single Agent RL
      2. Agent-Independent methods
      3. Agent-Tracking methods
      4. Agent-aware methods



## Introduction

​	Multi-agent reinforcement learning is a one of the most rapidly growing domain in machine learning. Yet, as  you have reached here there is relatively less formal textbook on this domain. So I have decided to summarise it from my point of view. Throughout this reviewing note, I would like to focus on two points below.

1. Stability of the agents' learning
2. Adaptation to the changing behaviour of the other agents

And also I will explain a variety of algorithms which support the evolution of multi-agent reinforcement learning. Indeed, the algorithms introduced so far can be visualised as below.



[table1: Breakdown of MARL algorithms by the type of task they address. From[1]](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/07_019.pdf)

![table1](https://github.com/Rowing0914/Reinforcement_Learning/blob/master/MARL/images/table1.PNG)



MARL stands on the various domains. For instance, as we will investigate more later, game theory forms the basement of most of algorithms in MARL. In fact, the most challenging part of MARL is the definition of the game setting. In this context, the game setting means the way to define the game and agents' behaviours. As a running example, I would like to recommend you to have a look at the Prisoner's Dilemma(Merrill Flood and Melvin Dresher in 1950). In the game, there are two players who have two actions, which is confess or stays silent respectively. The connection between Game theory and Reinforcement learning exists at this point. Game theory allows us to formally and radically state the game. And Reinforcement Learning defines the learning methods for the agents to approach to the optimal point. Again, the optimal points do depend on the game settings. By the way, I strongly recommend you to skim through the introductory material ([Intro to Multiagent Reinforcement Learning](https://github.com/Rowing0914/Reinforcement_Learning/blob/master/MARL/review_intro_ppt/README.md)) to understand the basic of MARL.





## Materials

1. [Intro to Multiagent Reinforcement Learning](https://github.com/Rowing0914/Reinforcement_Learning/blob/master/MARL/review_intro_ppt/README.md)



## References

1. [Bus¸oniu, L., Babuska, R., De Schutter, B.: A comprehensive survey of multi-agent reinforcement
   learning. IEEE Transactions on Systems, Man, and Cybernetics. Part C: Applications
   and Reviews 38(2), 156–172 (2008)](http://www.dcsc.tudelft.nl/~bdeschutter/pub/rep/07_019.pdf)