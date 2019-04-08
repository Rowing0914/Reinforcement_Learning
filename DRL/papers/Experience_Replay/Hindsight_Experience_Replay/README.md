## Main Paper

- [Hindsight Experience Replay by M.Andrychowicz et al., NIPS 2017](<https://arxiv.org/pdf/1707.01495.pdf>)

- project page: <https://sites.google.com/site/hindsightexperiencereplay/>



## Summary

### Problem statement

- Dealing with **sparse rewards** is one of the biggest challenges in Reinforcement Learning (RL).
- a common challenge, especially for robotics, is the need to engineer a reward function that not only reflects the task at hand but is also carefully shaped to guide the policy optimisation
- RL is still lacking of an ability to reason the cause of failure in a task



### Idea of HER

The pivotal idea behind HER is to replay each episode with a different goal than the one the agent was trying to achieve, e.g. one of the goals which was achieved in the episode.



### Prior work

- [DQN]()
- [DPG](https://github.com/Rowing0914/Reinforcement_Learning/tree/master/DRL/papers/DPG)
- [DDPG](https://github.com/Rowing0914/Reinforcement_Learning/tree/master/DRL/papers/DDPG)
- UFA

### Proposition



## Algorithm

The idea behind Hindsight Experience Replay (HER) is very simple: after experiencing some episode
$s_0, s_1, . . . , s_T$ we store in the replay buffer every transition $s_t \rightarrow s_{t+1}$ not only with the original
goal used for this episode but also with a subset of other goals.

![](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/Experience_Replay/Hindsight_Experience_Replay/images/algorithm.png)



### Experiments

![](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/Experience_Replay/Hindsight_Experience_Replay/images/experiemnts.png)

#### Trick they used



#### Metrics





## Result

![](/home/noio0925/Desktop/research/Reinforcement_Learning/DRL/papers/Experience_Replay/Hindsight_Experience_Replay/images/result.png)



### Conclusion

