## URL

http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-1.pdf



## Notes

### Course logistics

- Course website: http://rail.eecs.berkeley.edu/deeprlcourse
- Piazza: UC Berkeley, CS294-112
- Subreddit (for non-enrolled students): www.reddit.com/r/berkeleydeeprlcourse/



### What is reinforcement learning, and why should we care?

- How do we build intelligen machines?
- Intelligent machine must be able to adapt
- DL helps us handle unstructured environments
- RL provides a formalis for behaviour
- What does end-to-end learning mean for sequential decision making? like, some applications in robotics
- Why should we study this now?
  - advances in DL
  - advances in RL
  - advances in computational capability
- Beyond learning from reward
  - basic RL deals with maximising rewards
  - this is not the only problem that matters for sequential decision making
    - learning reward functions from example
    - transferring knowledge between domains
    - learnig to predict and using prediction to act
- Are there other forms of supervision?
  - Learning from demonstrations
    - directly copying observed behaviour
    - inferring rewards from observed behaviour
  - learning from observing the world
    - learning to predict
    - unsupervised learning
  - learning from other tasks
    - transfer learning
    - meta-learning: learning to learn
- Linking RL to the human brain: [Reinforcement learning in the brain by Yael Niv, 2009](https://www.princeton.edu/~yael/Publications/Niv2009.pdf)
- Open problems in RL
  - Humans can learn incredibly quickly: RL methods are usually slow
  - Humans can reuse past knowledge in other domains
  - Not clear the definition of the reward function
  - Not clear the definition of the role of prediction in decision making