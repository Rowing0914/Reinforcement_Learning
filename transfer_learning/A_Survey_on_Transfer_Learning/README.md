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

In this context, *different among domain(source and target)* means that either

 	1. a label space $Y$ is different between domains
 	2. the conditional probability distributions between the domains are different

### A categorisation of Transfer Learning Techniques

1. **What to transfer**

   which part of knowledge can be transferred across domains or tasks. Some knowledge is specific for individual domains or tasks, and some knowledge may be common between different domains such that they may help improve performance for the target domain or task. **p4**

2. **How to transfer**

   After discovering which knowledge can be transferred, learning algorithms need to be developed to transfer the knowledge, which corresponds to the “how to transfer”issue. **p4**

3. **When to transfer**

   which situations, transferring skills should be done. Likewise, we are interested in knowing in which situations, knowledge should not be transferred.  **p4**

##### Taxonomy of Transfer Learning Situations

1. **Inductive Transfer Learning** setting

   the target task is different from the source task, no matter when the source and target domains are the same or not. So, some labelled data in the target domain are required to induce an objective predictive model for use in the target domain. Indeed, we can further categorise this domain into two different fundamental setting.

   - A lot of labelled data in the source domain are available
   - No labelled data in the source domain are available

2. **Transductive Transfer Learning** setting

   the source and target tasks are the same, while the source and target domains are different. no labelled data in the target domain are available while a lot of labelled data in the source domain are available. Indeed, we can further categorise this domain into two different fundamental setting.

   - The feature spaces between the source and target domains are different
   - The feature spaces between domains are the same

3. **Unsupervised Transfer Learning** setting

   the target task is different from but related to the source task. However, the unsupervised transfer learning focus on solving unsupervised learning tasks in the target domain, such as clustering, dimensionality reduction, and density estimation

   Approaches to transfer learning in the above three different settings can be summarised into four cases based on “What to transfer.” **p5**

   1. **Instance transfer**: to re-weight some labelled data in the source domain for use in the target domain **p5**

      [24] B. Zadrozny, “Learning and Evaluating Classifiers under Sample Selection Bias,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

      [28] W. Dai, G. Xue, Q. Yang, and Y. Yu, “Transferring Naive Bayes Classifiers for Text Classification,” Proc. 22nd Assoc. for the Advancement of Artificial Intelligence (AAAI) Conf. Artificial Intelligence, pp. 540-545, July 2007.
      [29] J. Quionero-Candela, M. Sugiyama, A. Schwaighofer, and N.D. Lawrence, Dataset Shift in Machine Learning. MIT Press, 2009.

      [30] J. Jiang and C. Zhai, “Instance Weighting for Domain Adaptation in NLP,” Proc. 45th Ann. Meeting of the Assoc. Computational Linguistics, pp. 264-271, June 2007. 

      [31] X. Liao, Y. Xue, and L. Carin, “Logistic Regression with an Auxiliary Data Source,” Proc. 21st Int’l Conf. Machine Learning, pp. 505-512, Aug. 2005. 

      [32] J. Huang, A. Smola, A. Gretton, K.M. Borgwardt, and B. Scho¨lkopf, “Correcting Sample Selection Bias by Unlabeled Data,” Proc. 19th Ann. Conf. Neural Information Processing Systems, 2007. 

      [33] S. Bickel, M. Bru¨ ckner, and T. Scheffer, “Discriminative Learning for Differing Training and Test Distributions,” Proc. 24th Int’l Conf. Machine Learning, pp. 81-88, 2007. 

      [34] M. Sugiyama, S. Nakajima, H. Kashima, P.V. Buenau, and M. Kawanabe, “Direct Importance Estimation with Model Selection and its Application to Covariate Shift Adaptation,” Proc. 20th Ann. Conf. Neural Information Processing Systems, Dec. 2008. 

      [35] W. Fan, I. Davidson, B. Zadrozny, and P.S. Yu, “An Improved Categorization of Classifier’s Sensitivity on Sample Selection Bias,” Proc. Fifth IEEE Int’l Conf. Data Mining, 2005.

   2. **Feature representation transfer**: find a "god" feature representation that reduces difference between the source and the target domains and the error of classification and regression models **p5**

      [8] J. Blitzer, M. Dredze, and F. Pereira, “Biographies, Bollywood, Boom-Boxes and Blenders: Domain Adaptation for Sentiment Classification,” Proc. 45th Ann. Meeting of the Assoc. Computational Linguistics, pp. 432-439, 2007. 

      [22] R. Raina, A. Battle, H. Lee, B. Packer, and A.Y. Ng, “Self-TaughtLearning: Transfer Learning from Unlabeled Data,” Proc. 24th Int’l Conf. Machine Learning, pp. 759-766, June 2007

      [36] W. Dai, G. Xue, Q. Yang, and Y. Yu, “Co-Clustering Based Classification for Out-of-Domain Documents,” Proc. 13th ACM SIGKDD Int’l Conf. Knowledge Discovery and Data Mining, Aug. 2007.  

      [37] R.K. Ando and T. Zhang, “A High-Performance Semi-Supervised Learning Method for Text Chunking,” Proc. 43rd Ann. Meeting on Assoc. for Computational Linguistics, pp. 1-9, 2005. 

      [38] J. Blitzer, R. McDonald, and F. Pereira, “Domain Adaptation with Structural Correspondence Learning,” Proc. Conf. Empirical Methods in Natural Language, pp. 120 128, July 2006.

      [39] H. Daume´ III, “Frustratingly Easy Domain Adaptation,” Proc. 45th Ann. Meeting of the Assoc. Computational Linguistics, pp. 256-263, June 2007.

      [40] A. Argyriou, T. Evgeniou, and M. Pontil, “Multi-Task Feature Learning,” Proc. 19th Ann. Conf. Neural Information Processing Systems, pp. 41-48, Dec. 2007.

      [41] A. Argyriou, C.A. Micchelli, M. Pontil, and Y. Ying, “A Spectral Regularization Framework for Multi-Task Structure Learning,” Proc. 20th Ann. Conf. Neural Information Processing Systems, pp. 25- 32, 2008.

      [42] S.I. Lee, V. Chatalbashev, D. Vickrey, and D. Koller, “Learning a Meta-Level Prior for Feature Relevance from Multiple Related Tasks,” Proc. 24th Int’l Conf. Machine Learning, pp. 489-496, July 2007.

      [43] T. Jebara, “Multi-Task Feature and Kernel Selection for SVMs,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

      [44] C. Wang and S. Mahadevan, “Manifold Alignment Using Procrustes Analysis,” Proc. 25th Int’l Conf. Machine Learning ,pp. 1120-1127, July 2008.

   3. **Parameter transfer**: discover shared parameters or priors between the source and the target domain models, which can benefit for transfer learning **p5**

      [45] N.D. Lawrence and J.C. Platt, “Learning to Learn with the Informative Vector Machine,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

      [46] E. Bonilla, K.M. Chai, and C. Williams, “Multi-Task Gaussian Process Prediction,” Proc. 20th Ann. Conf. Neural Information Processing Systems, pp. 153-160, 2008.

      [47] A. Schwaighofer, V. Tresp, and K. Yu, “Learning Gaussian Process Kernels via Hierarchical Bayes,” Proc. 17th Ann. Conf. Neural Information Processing Systems, pp. 1209-1216, 2005. 

      [48] T. Evgeniou and M. Pontil, “Regularized Multi-Task Learning,” Proc. 10th ACM SIGKDD Int’l Conf. Knowledge Discovery and Data Mining, pp. 109-117, Aug. 2004. 

      [49] J. Gao, W. Fan, J. Jiang, and J. Han, “Knowledge Transfer via Multiple Model Local Structure Mapping,” Proc. 14th ACM SIGKDD Int’l Conf. Knowledge Discovery and Data Mining, pp. 283-291, Aug. 2008.

   4. **Relational knowledge transfer**: build mapping of relational knowledge between the source and the target domains. Both domains are relational domains and i.i.d assumption is relaxed in each domain **p5**

      [50] L. Mihalkova, T. Huynh, and R.J. Mooney, “Mapping and Revising Markov Logic Networks for Transfer Learning,” Proc. 22nd Assoc. for the Advancement of Artificial Intelligence (AAAI) Conf. Artificial Intelligence, pp. 608-614, July 2007. 

      [51] L. Mihalkova and R.J. Mooney, “Transfer Learning by Mapping with Minimal Target Data,” Proc. Assoc. for the Advancement of Artificial Intelligence (AAAI ’08) Workshop Transfer Learning for Complex Tasks, July 2008. 

      [52] J. Davis and P. Domingos, “Deep Transfer via Second-Order Markov Logic,” Proc. Assoc. for the Advancement of Artificial Intelligence (AAAI ’08) Workshop Transfer Learning for Complex Tasks, July 2008.

We will look into all three problems settings mentioned above from now on.

### Inductive Transfer Learning

#### Definition 2 (Inductive Transfer Learning). 

Given a source domain $D_S$ and a learning task $T_S$, a target domain $D_T$ and a learning task $T_T$, inductive transfer learning aims to help improve the learning of the target predictive function $f(cdot)$ in $D_T$ using the knowledge in $D_S$ and $T_S$, where $T_S \neq T_T$.  **p6**

So, put it differently, a few labelled data in the target domain are required as the training data to induce the target predictive function.  **p6**

#### Transferring Knowledge of Instances(instance-transfer approach)

- Dai et al. [6] proposed a boosting algorithm, **TrAdaBoost**, which is an extension of the *AdaBoost* algorithm, to address the inductive transfer learning problems. **TrAdaBoost** assumes that the source and target-domain data use exactly the same set of features and labels, but the distributions of the data in the two domains are different.  **p6**

  [6] W. Dai, Q. Yang, G. Xue, and Y. Yu, “Boosting for Transfer Learning,” Proc. 24th Int’l Conf. Machine Learning, pp. 193-200, June 2007.

- Jiang and Zhai [30] proposed a heuristic method to remove “misleading” training examples from the source domain based on the difference between conditional probabilities $P(y_T | x_T)$and $P(y_S | x_S)$ .  **p6**

  [30] J. Jiang and C. Zhai, “Instance Weighting for Domain Adaptation in NLP,” Proc. 45th Ann. Meeting of the Assoc. Computational Linguistics, pp. 264-271, June 2007. 

- Liao et al. [31] proposed a new active learning method to select the unlabelled data in a target domain to be labelled with the help of the source domain data.   **p6**

  [31] X. Liao, Y. Xue, and L. Carin, “Logistic Regression with an Auxiliary Data Source,” Proc. 21st Int’l Conf. Machine Learning, pp. 505-512, Aug. 2005. 

- Wu and Dietterich [53] integrated the source domain (auxiliary) data an Support Vector Machine (SVM) framework for improving the classification performance.  **p6**

  [53] P. Wu and T.G. Dietterich, “Improving SVM Accuracy by Training on Auxiliary Data Sources,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

#### Transferring Knowledge of Feature Representations

The feature-representation-transfer approach to the inductive transfer learning problem aims at finding “good” feature representations to minimise domain divergence and classification or regression model error. Strategies to find “good” feature representations are different for different types of the source domain data. **p6**

##### Supervised Feature Construction

The basic idea is to learn a low-dimensional representation that is shared across related tasks. In addition, the learned new representation can reduce the classification or regression model error of each task as well.  **p6**

- Argyriou et al. [40] proposed a sparse feature learning method for multitask learning. In the inductive transfer learning setting, the common features can be learned by solving an optimisation problem.  **p6**

  [40] A. Argyriou, T. Evgeniou, and M. Pontil, “Multi-Task Feature Learning,” Proc. 19th Ann. Conf. Neural Information Processing Systems, pp. 41-48, Dec. 2007.

- In a follow-up work, Argyriou et al. [41] proposed a spectral regularisation framework on matrices for multitask structure learning.  **p6**

  [41] A. Argyriou, C.A. Micchelli, M. Pontil, and Y. Ying, “A Spectral Regularisation Framework for Multi-Task Structure Learning,” Proc. 20th Ann. Conf. Neural Information Processing Systems, pp. 25- 32, 2008.

- Lee et al. [42] proposed a convex optimisation algorithm for simultaneously learning meta-priors and feature weights from an ensemble of related prediction tasks. The meta-priors can be transferred among different tasks.  **p6**

  [42] S.I. Lee, V. Chatalbashev, D. Vickrey, and D. Koller, “Learning a Meta-Level Prior for Feature Relevance from Multiple Related Tasks,” Proc. 24th Int’l Conf. Machine Learning, pp. 489-496, July 2007.

- Jebara [43] proposed to select features for multitask learning with SVMs.   **p6**

  [43] T. Jebara, “Multi-Task Feature and Kernel Selection for SVMs,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

- Ruckert and Kramer [54] designed a kernel-based approach to inductive transfer, which aims at finding a suitable kernel for the target data.  **p6**

  [54] U. Ruckert and S. Kramer, “Kernel-Based Inductive Transfer,” Proc. European Conf. Machine Learning and Knowledge Discovery in Databases (ECML/PKDD ’08), pp. 220-233, Sept. 2008.

##### Unsupervised Feature Construction

- In [22], Raina et al. proposed to apply sparse coding [55], which is an unsupervised feature construction method, for learning higher level features for transfer learning.   **p6**

  [22] R. Raina, A. Battle, H. Lee, B. Packer, and A.Y. Ng, “Self-TaughtLearning: Transfer Learning from Unlabeled Data,” Proc. 24th Int’l Conf. Machine Learning, pp. 759-766, June 2007

  [55] H. Lee, A. Battle, R. Raina, and A.Y. Ng, “Efficient Sparse Coding
  Algorithms,” Proc. 19th Ann. Conf. Neural Information Processing
  Systems, pp. 801-808, 2007.

- Recently, manifold learning methods have been adapted for transfer learning. In [44], Wang and Mahadevan proposed a Procrustes analysis-based approach to manifold alignment without correspondences, which can be used to transfer the knowledge across domains via the aligned manifolds.  **p6**

  [44] C. Wang and S. Mahadevan, “Manifold Alignment Using Procrustes Analysis,” Proc. 25th Int’l Conf. Machine Learning ,pp. 1120-1127, July 2008.

#### Transferring Knowledge of Parameters

 Most parameter-transfer approaches to the inductive transfer learning setting assume that individual models for related tasks should share some parameters or prior distributions of hyper-parameters. **p7**

- Lawrence and Platt [45] proposed an efficient algorithm known as MT-IVM, which is based on Gaussian Processes (GP), to handle the multitask learning case. MT-IVM tries to learn parameters of a Gaussian Process over multiple tasks by sharing the same GP prior.

  [45] N.D. Lawrence and J.C. Platt, “Learning to Learn with the Informative Vector Machine,” Proc. 21st Int’l Conf. Machine Learning, July 2004.

- Bonilla et al. [46] also investigated multitask learning in the context of GP. The authors proposed to use a free-form covariance matrix over tasks to model intertask dependencies, where a GP prior is used to induce correlations between tasks.

  [46] E. Bonilla, K.M. Chai, and C. Williams, “Multi-Task Gaussian Process Prediction,” Proc. 20th Ann. Conf. Neural Information Processing Systems, pp. 153-160, 2008.

- Schwaighofer et al. [47] proposed to use a hierarchical Bayesian framework (HB) together with GP for multitask learning.

  [47] A. Schwaighofer, V. Tresp, and K. Yu, “Learning Gaussian Process Kernels via Hierarchical Bayes,” Proc. 17th Ann. Conf. Neural Information Processing Systems, pp. 1209-1216, 2005. 

- Evgeniou and Pontil [48] borrowed the idea of HB to SVMs for multitask learning. And the base idea is that transferring the priors of the GP models, some researchers also proposed to transfer parameters of SVMs under a regularisation framework

  [48] T. Evgeniou and M. Pontil, “Regularized Multi-Task Learning,” Proc. 10th ACM SIGKDD Int’l Conf. Knowledge Discovery and Data Mining, pp. 109-117, Aug. 2004. 
