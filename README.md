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
## Motivation & Problem Setup
- The framework we study in this paper is the _stochastic approximation (SA)_ framework, which captures a large class of problems in optimization, control and RL such as SGD, TD-learning and Q-learning.
- We wish to speed up the learning process by collecting data collaboratively from different environments, taking inspiration from federated learning (FL).
- However, in practice, data usually come from different (heterogeneous) environments, and there is ample reason to consider correlation between data samples, which is the case in practical RL algorithms.
- The problem interest is then
  ![image](https://github.com/user-attachments/assets/ed74cfbf-d38b-466f-a17e-c1dbcc473c54)
  - Agent $i$ only has access to a sequence of noisy operators $G_i(\cdot, o_{i,t})$.
  - The observations ${o_{i,t}}$ are genearted from an ergodic Markov chain specific to agent $i$ to capture temporal correlations between data.
  - Agents communicate intermittenly through a central server.
### Assumptions
![image](https://github.com/user-attachments/assets/bfdba82a-ae7b-475a-9427-84d70b4eeefb)
![image](https://github.com/user-attachments/assets/cebcdd0b-63d8-426a-b3e1-fe1d6c561902)
![image](https://github.com/user-attachments/assets/5ed8f99c-ceba-498f-842e-276c5eb2398f)
## Why We Need A New Algorithm?
**Even without noise, as long as the number of local updates is strictly greater than 1, there exists a heterogeneity bias term that impedes the convergence.**
![image](https://github.com/user-attachments/assets/4204b6d2-ae7a-488f-a40e-1248f79e0b18)
## FedHSA-Federated Heterogeneous Stochastic Approximation
- Same idea as **Fast-FedPG**: Adding a correction term in every local update
![image](https://github.com/user-attachments/assets/2385b3f0-c6e3-4704-ac0f-e8ed0dab7107)
## Theoretical Guarantees
![image](https://github.com/user-attachments/assets/c76c2250-2e6c-4962-bc22-c1d65765a6ad)
- Linear speedup
- No heterogeneity bias
- Matching centralized rates
## Challenges in Proof
![image](https://github.com/user-attachments/assets/aa557ef7-5ae4-4172-b702-ec1fb6060024)
- This analysis is different from Fast-FedPG since there is data correlation interleaved with heterogeneity, which poses as a great challenge.
## Simulation Results
![image](https://github.com/user-attachments/assets/9981af47-cbf6-4629-8dce-1a634711724e)
- FedHSA has a lower error floor compared to the vanilla federated SA algorithm.
- The error floor improves with even more agents.







  
