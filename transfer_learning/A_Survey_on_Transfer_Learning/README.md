## Profile of paper

Title: A Survey on Transfer Learning

Authors: Sinno Jialin Pan and Qiang Yang, Fellow, IEEE



## Paper Structure

- Introduction
- Overview

  - A brief history of transfer learning
  - Notations and Definitions
  - A categorisation of transfer learning techniques
- Inductive Transfer Learning
  - Transferring knowledge of instances
  - Transferring knowledge of features representations
    - supervised feature construction
    - unsupervised feature construction
  - Transferring knowledge of parameters
  - Transferring relational knowledge
- Transductive Transfer Learning
  - Transferring knowledge of instances
  - Transferring knowledge of features representations
  - Unsupervised Transfer learning
  - Transfer bounds and negative transfer
- Applications of Transfer learning
- Conclusions



## Note

### A Brief History of Transfer learning

#### Problem Setting in general

We sometimes have a classification task in one domain of interest, but we only have sufficient training data in another domain of interest, where the latter data may be in a different feature space or follow a different data distribution. In such cases, **knowledge transfer**, if done successfully, would greatly improve the performance of learning by avoiding much expensive data-labelling efforts. 

#### Problem in traditional machine learning algorithms

Traditional data mining and machine learning algorithms make predictions on the future data using statistical models that are trained on previously collected labelled or unlabelled training data.

[11] X. Yin, J. Han, J. Yang, and P.S. Yu, “Efficient Classification across Multiple Database Relations: A Crossmine Approach,” IEEE Trans. Knowledge and Data Eng., vol. 18, no. 6, pp. 770-783, June 2006.
[12] L.I. Kuncheva and J.J. Rodrłguez, “Classifier Ensembles with a Random Linear Oracle,” IEEE Trans. Knowledge and Data Eng., vol. 19, no. 4, pp. 500-508, Apr. 2007.
[13] E. Baralis, S. Chiusano, and P. Garza, “A Lazy Approach to Associative Classification,” IEEE Trans. Knowledge and Data Eng., vol. 20, no. 2, pp. 156-171, Feb. 2008.

#### Motivation for *transfer learning*

The study of Transfer learning is motivated by the fact that people can intelligently apply knowledge learned previously to solve new problems faster or with better solutions. The fundamental motivation for Transfer learning in the field of machine learning was discussed in a NIPS-95 workshop on “Learning to Learn”, which focused on the need for lifelong machine learning methods that retain and reuse previously learned knowledge. **p2**

```latex
====== NIPS-95 workshop on “Learning to Learn” ======
@book{thrun2012learning,
  title={Learning to learn},
  author={Thrun, Sebastian and Pratt, Lorien},
  year={2012},
  publisher={Springer Science \& Business Media}
}
```

#### Similarity and Dissimilarity: transfer learning vs multi-task learning

**1. Similarity**

Among these, a closely related learning technique to transfer learning is the multitask learning framework [21], which tries to learn multiple tasks simultaneously even when they are different. A typical approach for multitask learning is to uncover the common (latent) features that can benefit each individual task. **p2** 

[21] R. Caruana, “Multitask Learning,” Machine Learning, vol. 28, no. 1, pp. 41-75, 1997.

**2. Dissimilarity**

In 2005, the Broad Agency Announcement (BAA) 05-29 of Defense Advanced Research Projects Agency (DARPA)’s Information Processing Technology Office (IPTO) gave a new mission of transfer learning: the ability of a system to recognise and apply knowledge and skills learned in previous tasks to novel tasks. In this definition, transfer learning aims to extract the knowledge from one or more source tasks and applies the knowledge to a target task. In contrast to multitask learning, rather than learning all of the source and target tasks simultaneously, transfer learning cares most about the target task. The roles of the source and target tasks are no longer symmetric in transfer learning. **p2**

### Notations and Definitions

#### Domain and Task

- *a domain* consists of two components: a feature space $\chi$ and a marginal probability distribution $P(X)$, where $X = \{ x_1, x_2, \dots, x_n \} \in \chi$. In general, if two domains are different, then they may have different feature spaces or different marginal probability distributions. **p2-3**

- given a specific domain defined as before, $D = \{ \chi, P(X) \}$, *a task* consists of two components as it is in the normal machine learning settings: a label space $y$ and an objective predictive function $f(\cdot)$. Hence *a task* can be formulated as: $T = \{ y, f(\cdot) \}$. In addition, we define the training data, which consists of pairs $\{ x_i, y_i \}$, where $x_i \in X$ and $y_i \in Y$ **p3**

  ##### Definition 1(Transfer Learning)

  Given a source domain $D_S$ and learning task $T_S$, a target domain $D_T$ and learning task $T_T$ , transfer learning aims to help improve the learning of the target predictive function $f_T(\cdot)$ in $D_T$ using the knowledge in $D_S$ and $T_S$, where $D_S \neq D_T$ , or $T_S \neq T_T$.    **p3**