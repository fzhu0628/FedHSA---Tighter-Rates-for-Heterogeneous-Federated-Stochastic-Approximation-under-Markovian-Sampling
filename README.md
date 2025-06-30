# Achieving Tighter Finite-Time Rates for Heterogeneous Federated Stochastic Approximation under Markovian Sampling
## Overview
- This is my second paper for the pursuit of my PhD degree.
- This paper has been submitted to NeuRIPS 2025.
- About this paper
  - Two sets of algorithms in reinforcement learning (RL): _policy iteration_, and _value-iteration_.
  - Recall that in my last paper _Towards Fast Rates for Federated and Multi Task Reinforcement Learning_, we focused on heterogeneous FRL problems with _polcy-iteration_ methods, policy gradient methods to be precise.
  - We propose _**FedHSA**_, an algorithm designed for _value-iteration_ methods in heterogeneous FRL settings, compensenting for the previous paper in FRL algorithms.
  - We consider the generic federated _stochastic approximation_ framework, where the **goal** is for the agents to _communicate intermittently_ and find the _fixed point_ of an average of _different_ contractive operators. This formulation can be applied to many RL algorithms such as TD and Q.
  - The proposed FedHSA algorithm is proved to converge precisely to the desired point with _collaborative speedup_ and **no heterogeneity bias** under _Markovian sampling_.
  - The Markovian data and heterogeneous operators account for its applicability to FRL algorithms.
