# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:05:05 2025

@author: fzhu5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def markov_gen(S, gamma, r, het):
    # Generate a Markov Matrix
    P = torch.rand(S, S, device=device) + torch.rand(S, S, device=device) * het
    P = P / P.sum(dim=1, keepdims=True)  # Normalize rows to sum to 1

    # Find the eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = torch.linalg.eig(P.T)
    ind = torch.argmin(torch.abs(eigenvalues.real - 1))
    p = eigenvectors[:, ind].real
    p = p * torch.sign(p)
    p = p / p.sum()  # Stationary distribution

    D = torch.diag(p)  # Diagonal matrix of stationary distribution

    R = 2 * torch.rand(S, 1, device=device)  # Reward vector
    # noise = np.random.exponential(scale = het, size = (S, 1))
    
    # R += noise

    # Generating the feature matrix
    phi = torch.zeros((S, r), device=device)
    for i in range(min(S, r)):
        phi[i, i] = 1
        
    # np.random.seed(43)
    # phi = np.random.rand(S, r)

    # Proj = phi @ np.linalg.inv(phi.T @ D @ phi) @ phi.T @ D
    # A = (np.eye(S) - gamma * Proj @ P) @ phi
    # b = Proj @ R
    A = phi.T @ D @ (gamma * P @ phi - phi)
    b = - phi.T @ D @ R
    theta_t = torch.linalg.pinv(A) @ b  # Fixed point of TD

    return theta_t, P, R, p, phi, A, b

# Parameters
S = 100  # Number of states
r = 100  # Rank of feature matrix
T = 100  # Number of iterations
H = 10 # Number of local steps
alpha = 0.1
num_agents = 10

P_all = []
R_all = []
p_all = []
phi_all = []
A_all = []
b_all = []
theta_st_all = []

gamma = torch.rand(num_agents, 1, device=device) * 0.49 + 0.5

phi_global = torch.rand(S, r, device=device)

for i in range(num_agents):
    # Generate Markovian data
    theta_st, P, R, p, phi, A, b = markov_gen(S, gamma[i], r, 1)
    
    # phi = phi_global.copy()
    
    P_all.append(P)
    R_all.append(R)
    p_all.append(p)
    phi_all.append(phi)
    A_all.append(A)
    b_all.append(b)
    theta_st_all.append(theta_st)

A_global = torch.stack(A_all).sum(dim=0)
b_global = torch.stack(b_all).sum(dim=0)
# theta_true = np.linalg.solve(A_global, b_global)  # Fixed point of TD
theta_true = torch.linalg.pinv(A_global) @ b_global  # Fixed point of TD

x_global_initial = torch.randn(r, 1, device=device)

# %% vanilla local SGD
norm_differences = []
states = torch.zeros(num_agents, device=device, dtype=torch.long)

x_global = x_global_initial.clone()
for _ in range(T):
    x_local_list = []    
    for i in range(num_agents):
        x_local = x_global.clone()
        for _ in range(H):
            s_old = states[i].item()
            s_new = torch.multinomial(P_all[i][s_old], num_samples=1).item()
            # s_new = np.random.choice(S, p=p_all[i])
            states[i] = s_new
            
            rew = R_all[i][s_old, 0]
            g = (rew + gamma[i] * phi_all[i][states[i], :] @ x_local - phi_all[i][s_old, :] @ x_local) * phi_all[i][s_old, :]
            
            x_local += alpha * g.unsqueeze(1)
            
        x_local_list.append(x_local)
        
    x_global = torch.mean(torch.stack(x_local_list), dim=0)
    
    norm_difference = torch.norm(x_global - theta_true).item() ** 2
    norm_differences.append(norm_difference)

#%% FedHSA
norm_differences1 = []
states = torch.zeros(num_agents, device=device, dtype=torch.long)

x_global = x_global_initial.clone()
for _ in range(T):
    x_local_list = []    
    g_begin_global = torch.zeros(r, device=device)
    g_begin_list = []
    for i in range(num_agents):
        s_old = states[i].item()
        s_new = torch.multinomial(P_all[i][s_old], num_samples=1).item()
        # s_new = np.random.choice(S, p=p_all[i])
        states[i] = s_new
        
        g_begin = (R_all[i][s_old, 0] + gamma[i] * phi_all[i][s_new, :] @ x_global - phi_all[i][s_old, :] @ x_global) * phi_all[i][s_old, :]
        g_begin_list.append(g_begin)
        g_begin_global += g_begin
    g_begin_global /= num_agents
    for i in range(num_agents):
        x_local = x_global.clone()
        for step in range(H):
            if step == 0:
                g = g_begin_list[i]
            else:
                s_old = states[i].item()
                s_new = torch.multinomial(P_all[i][s_old], num_samples=1).item()
                # s_new = np.random.choice(S, p=p_all[i])
                states[i] = s_new
                
                rew = R_all[i][s_old, 0]
                g = (rew + gamma[i] * phi_all[i][states[i], :] @ x_local - phi_all[i][s_old, :] @ x_local) * phi_all[i][s_old, :]
            
            if step == 0:
                g_begin_local = g
            
            x_local += alpha * (g + g_begin_global - g_begin_local).unsqueeze(1)
            # x_local += alpha * g.reshape(-1, 1)
            
        x_local_list.append(x_local)
        
    x_global = torch.mean(torch.stack(x_local_list), dim=0)
    
    norm_difference = torch.norm(x_global - theta_true).item() ** 2
    norm_differences1.append(norm_difference)
#%%
plt.figure()
plt.semilogy(norm_differences, linewidth=1, label = "Vanilla Federated TD")
plt.semilogy(norm_differences1, linewidth=1, label = "FedHSA")
plt.xlim([1, T])
plt.xlabel("Round")
plt.ylabel(r'$||{\bar\theta}^{(t)}-\theta^\star||^2$')
plt.title("Comparison between vanilla federated TD and FedHSA")
plt.legend()
plt.grid(True)
# plt.savefig("comp_vanilla_noiseless_TD.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()
