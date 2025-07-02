# Achieving Tighter Finite-Time Rates for Heterogeneous Federated Stochastic Approximation under Markovian Sampling
## Overview
- This is my second paper for the pursuit of my PhD degree.
- This paper has been submitted to NeuRIPS 2025.
- About this paper
  - Two sets of algorithms in reinforcement learning (RL): _policy iteration_, and _value-iteration_.
  - Recall that in my last paper _Towards Fast Rates for Federated and Multi Task Reinforcement Learning_, we focused on heterogeneous FRL problems with _polcy-iteration_ methods, policy gradient methods to be precise.
  - We propose _**FedHSA**_, an algorithm designed for _value-iteration_ methods in heterogeneous FRL settings, compensenting for the previous paper in FRL algorithms.
  - We consider the generic federated _stochastic approximation_ framework, where the **goal** is for the agents to _communicate intermittently_ via a server and find the _fixed point_ of an average of _different_ contractive operators. This formulation can be applied to many RL algorithms such as TD and Q.
  - The proposed FedHSA algorithm is proved to converge precisely to the desired point with _collaborative speedup_ and **no heterogeneity bias** under _Markovian sampling_.
  - The Markovian data and heterogeneous operators account for its applicability to FRL algorithms.
## Motivation
- The framework we study in this paper is the _stochastic approximation (SA)_ framework, which captures a large class of problems in optimization, control and RL such as SGD, TD-learning and Q-learning.
- We wish to speed up the learning process by collecting data collaboratively from different environments, taking inspiration from federated learning (FL).
- However, in practice, data usually come from different (heterogeneous) environments, and there is ample reason to consider correlation between data samples, which is the case in practical RL algorithms.
## Problem Setup
- The problem interest is

![image](https://github.com/user-attachments/assets/177fb1be-80c8-4476-8e56-03739700b586)
  - Agent $i$ only has access to a sequence of noisy operators $G_i(\cdot, o_{i,t})$.
  - The observations $\{o_{i,t}\}$ are genearted from an ergodic Markov chain specific to agent $i$ to capture temporal correlations between data.
  - Agents communicate intermittenly through a central server.
  - In the single-agent (centralized) setting, athe finite-time rate for SA is

![image](https://github.com/user-attachments/assets/6d763e40-a47a-4f1e-95fc-69180a28d5f8)

where $d_t:=\mathbb{E}\left[\lVert\theta^{(t)}-\theta^\star\rVert^2\right]$ denotes the MSE at time-step $t$.
- **Key Question:** Is it possible to converge exactly to $\theta^\star$ in the federated case, with a $M$-fold reduction on the variance term, capturing the benefit of collaboration?

### Assumptions
- The local noisy operators $G_i$'s are $L$-Lipschitz.
- The average true operator $\bar G$ is 1-point strongly monotone w.r.t. $\theta^\star$.
- Each agent's underlying Markov chain is ergodic.
- Observations genearted for different agents are independent.

## Why We Need A New Algorithm?
- Even given that each agent has access to the true local operator $\bar G_i$, but the number of local steps $H=2$, we can prove that the vanilla FedAvg-like local SA algorithm fails to converge to the true optimum $\theta^\star$, known as the ''client-drift'' issue in federated optimization.
- To be specific, there is a bias term in the final bound that impedes convergence. _Can we get rid of that bias while still enjoying the benefit of collaboration?_

## FedHSA-Federated Heterogeneous Stochastic Approximation
- Same idea as **Fast-FedPG**: Adding a correction term in every local update
![image](https://github.com/user-attachments/assets/06d7fbb8-4af8-40f3-9519-e9c35f048ecc)

## Theoretical Guarantees
### Challenges in Proof
- Two types of data correlations have to be tackled: i) temporal correlations in the data for any given agent, and ii) correlations induced by exchanging data across agents---temporal and spatial correlations.

![image](https://github.com/user-attachments/assets/c8cefffa-9167-450f-83eb-874b278ff8ed)
### Main results
- Matching centralized rates with variance reduced $M$-fold.

![image](https://github.com/user-attachments/assets/d6921f56-af5f-47af-aac1-6fed4a967dcc)
- Linear speedup w.r.t. the number of agents with no heterogeneity bias.

![image](https://github.com/user-attachments/assets/9c221cc9-7cb4-4f7f-8fbc-9c12740fab9a)

## Simulation Results
![image](https://github.com/user-attachments/assets/9981af47-cbf6-4629-8dce-1a634711724e)
- FedHSA has a lower error floor compared to the vanilla federated SA algorithm.
- The error floor improves with even more agents.







  
