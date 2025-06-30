# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:05:05 2025

@author: fzhu5
"""

import numpy as np
import matplotlib.pyplot as plt

def markov_gen(S, gamma, r, het):
    # Generate a Markov Matrix
    P = np.random.rand(S, S) + np.random.rand(S, S) * het
    P = P / P.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1

    # Find the eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    ind = np.argmin(np.abs(eigenvalues - 1))
    p = np.real(eigenvectors[:, ind])
    p = p * np.sign(p)
    p = p / p.sum()  # Stationary distribution

    D = np.diag(p)  # Diagonal matrix of stationary distribution

    R = 2 * np.random.rand(S, 1)  # Reward vector
    # noise = np.random.exponential(scale = het, size = (S, 1))
    
    # R += noise

    # Generating the feature matrix
    # phi = np.zeros((S, r))
    # for i in range(min(S, r)):
    #     phi[i, i] = 1
        
    # np.random.seed(43)
    phi = np.random.rand(S, r)

    # Proj = phi @ np.linalg.inv(phi.T @ D @ phi) @ phi.T @ D
    # A = (np.eye(S) - gamma * Proj @ P) @ phi
    # b = Proj @ R
    A = phi.T @ D @ (gamma * P @ phi - phi)
    b = - phi.T @ D @ R
    theta_t = np.linalg.pinv(A) @ b  # Fixed point of TD

    return theta_t, P, R, p, phi, A, b

# Parameters
S = 100  # Number of states
r = 50  # Rank of feature matrix
T = 2000  # Number of iterations
H = 10 # Number of local steps
alpha = 0.03

number_list = [1, 5, 20, 100]
results = []

for num_agents in number_list:
    P_all = []
    R_all = []
    p_all = []
    phi_all = []
    A_all = []
    b_all = []
    theta_st_all = []
    
    gamma = np.random.uniform(0.5, 0.99, size=(num_agents, 1)) # varying gammas to increase heterogeneity
    # gamma = [0.7] * num_agents
    
    phi_global = np.random.rand(S, r)
    
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
    
    A_global = sum(A_all)
    b_global = sum(b_all)
    # theta_true = np.linalg.solve(A_global, b_global)  # Fixed point of TD
    theta_true = np.linalg.pinv(A_global) @ b_global  # Fixed point of TD
    
    x_global_initial = np.random.randn(r, 1)
    
#%% FedHSA
    norm_differences1 = []
    states = [0] * num_agents

    x_global = x_global_initial.copy()
    for _ in range(T):
        x_local_list = []    
        g_begin_global = np.zeros(r)
        g_begin_list = []
        for i in range(num_agents):
            s_old = states[i]
            s_new = np.random.choice(S, p=P_all[i][s_old])
            # s_new = np.random.choice(S, p=p_all[i])
            states[i] = s_new
            
            g_begin = (R_all[i][s_old, 0] + gamma[i] * phi_all[i][s_new, :] @ x_global - phi_all[i][s_old, :] @ x_global) * phi_all[i][s_old, :]
            g_begin_list.append(g_begin)
            g_begin_global += g_begin
        g_begin_global /= num_agents
        for i in range(num_agents):
            x_local = x_global.copy()
            for step in range(H):
                if step == 0:
                    g = g_begin_list[i]
                else:
                    s_old = states[i]
                    s_new = np.random.choice(S, p=P_all[i][s_old])
                    # s_new = np.random.choice(S, p=p_all[i])
                    states[i] = s_new
                    
                    rew = R_all[i][s_old, 0]
                    g = (rew + gamma[i] * phi_all[i][states[i], :] @ x_local - phi_all[i][s_old, :] @ x_local) * phi_all[i][s_old, :]
                
                if step == 0:
                    g_begin_local = g
                
                x_local += alpha * (g + g_begin_global - g_begin_local).reshape(-1, 1)
                # x_local += alpha * g.reshape(-1, 1)
                
            x_local_list.append(x_local)
            
        x_global = np.mean(x_local_list, axis=0)
        
        norm_difference = np.linalg.norm(x_global - theta_true)**2
        norm_differences1.append(norm_difference)
    results.append(norm_differences1)

#%%
# Plot the convergence behavior
plt.figure(figsize=(8, 5))
# plt.semilogx(results[0], label= str(number_list[0] ) + " agent")
# plt.semilogx(results[1], label= str(number_list[1] ) +" agents")
# plt.semilogx(results[2], label= str(number_list[2] ) +" agents")
# plt.semilogx(results[3], label= str(number_list[3] ) +" agents")
plt.plot(results[0], label= str(number_list[0] ) + " agent", color='b', linewidth=2, markevery=50)
plt.plot(results[1], '--o', label= str(number_list[1] ) +" agents", color='r', linewidth=2,markevery=50)
plt.plot(results[2],'-^', label= str(number_list[2] ) +" agents", color='g', linewidth=2,markevery=50)
plt.plot(results[3],'-*', label= str(number_list[3] ) +" agents", color=[17/255, 17/255, 17/255], linewidth=2,markevery=50)
plt.xlabel(r'${{t}}$', fontsize=30, fontname='Times New Roman', weight='bold')
plt.ylabel(r'$E_t$', fontsize=30, fontname='Times New Roman', weight='bold')
plt.yscale('log')
plt.legend(loc='best', fontsize=20)
# Grid
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
# Axes properties
# plt.tick_params(axis='both', which='major', labelsize=15)
# Set tick parameters
plt.tick_params(
    axis='both', which='major', width=2, length=10, labelsize=20, direction='in'
)
plt.tick_params(
    axis='both', which='minor', width=1, length=5, labelsize=20, direction='in'
)

plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Automatically adjust layout
plt.tight_layout()

plt.savefig("comp_TD_agents.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()
