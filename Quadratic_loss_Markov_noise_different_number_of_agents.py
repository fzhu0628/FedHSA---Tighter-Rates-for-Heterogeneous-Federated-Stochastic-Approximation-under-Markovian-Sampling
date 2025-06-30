# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:14:34 2024

@author: 10123
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_dim = 10       # Dimension of the optimization variable x
num_local_steps = 10  # Local steps before aggregation
heterogeneity_level = 0.05  # Controls how different the local loss functions are
noise_variance = 1e-0  # Variance of the gradient noise


threshold = 1e-20  # Convergence threshold
max_rounds = 1000  # Maximum number of rounds
learning_rate = 1e-2

target_smoothness = 10
target_convexity = 1

number_list = [1, 5, 20, 100]
results = []
for number in range(len(number_list)):
    num_agents = number_list[number]
# Federated optimization routine with stopping criterion
# Create Markov chains for gradient noise
    # Generate random local loss functions for agents
    A_matrices = []
    b_vectors = []

    for i in range(num_agents):
        # Generate symmetric matrix A_i and ensure positive definiteness after heterogeneity adjustment
        random_matrix = np.random.randn(n_dim, n_dim)
        A_i = np.dot(random_matrix.T, random_matrix) / n_dim  # Base symmetric matrix
        # A_i += heterogeneity_level * np.random.randn(n_dim, n_dim)
        # A_i = np.random.randn(n_dim, n_dim) / n_dim
        # if i != 1:
        #     A_i += heterogeneity_level * np.random.randn(n_dim, n_dim) # No need to add that for single agent
        A_i = (A_i + A_i.T) / 2  # Ensure symmetry
        A_i += np.eye(n_dim) * (np.max(-np.linalg.eigvalsh(A_i), 0) + 10)
        
        # # Get eigenvalues of A_i
        # eigenvalues = np.linalg.eigvalsh(A_i)
        # lambda_min = np.min(eigenvalues)
        # lambda_max = np.max(eigenvalues)
    
        # # Scale and shift eigenvalues to fit [μ, L]
        # A_i = ((target_smoothness - target_convexity) / (lambda_max - lambda_min)) * (A_i - lambda_min * np.eye(n_dim)) + target_convexity * np.eye(n_dim)
        A_matrices.append(A_i)

        # Generate random vector b_i
        b_i = np.random.randn(n_dim)
        # if i != 1:
        #     b_i += heterogeneity_level * np.random.randn(n_dim)  # Adjust for heterogeneity
        b_vectors.append(b_i)

    # Compute the analytical solution to the global optimization problem
    A_global = sum(A_matrices)
    b_global = sum(b_vectors)
    analytical_solution = np.linalg.solve(A_global, b_global)

    # Initialize global variable x
    x_global_initial = np.random.randn(n_dim)
    
    # Markovian noise
    markov_states = []
    transition_matrices = []
    
    for i in range(num_agents):
        # Create a transition matrix with rows summing to 1 and non-negative entries
        vector = np.random.randn(n_dim)  
        markov_states.append(vector)
        
        matrix = np.random.randn(n_dim, n_dim)
        matrix = matrix / np.max(np.abs(np.linalg.eigvals(matrix))) / 1.2  # Normalize spectral radius to < 1
        transition_matrices.append(matrix)
    
    norm_differences_FedHSA = []
    x_global = x_global_initial.copy()
    norm_differences_FedHSA.append(np.linalg.norm(x_global - analytical_solution)**2)
    for round in range(max_rounds):
        # IID noise
        # iid_noise = []
        
        # for i in range(num_agents):
        #     iid_noise.append(np.random.rand(n_dim) * noise_variance)

        x_local_list = []
        gradient_begin_global = np.dot(np.mean(A_matrices, axis=0), x_global) - np.mean(b_vectors, axis=0) + np.mean(markov_states, axis=0)
        # gradient_begin_global = np.dot(np.mean(A_matrices, axis=0), x_global) - np.mean(b_vectors, axis=0) + np.mean(iid_noise, axis=0)
        for i in range(num_agents):
            x_local = x_global.copy()
    
            # Perform local steps using gradient descent with correction
            for step in range(num_local_steps):
                gradient = np.dot(A_matrices[i], x_local) - b_vectors[i]
                
                # Add Markovian noise to the gradient
                noise = markov_states[i]
                gradient += noise
                # gradient += iid_noise[i]
    
                # Update Markov state
                markov_states[i] = np.dot(transition_matrices[i], markov_states[i]) + noise_variance * np.random.randn(n_dim)
                
                if step == 0:
                    gradient_begin_local = gradient.copy()
                
                if num_agents != 1:
                    x_local -= learning_rate * (gradient + gradient_begin_global - gradient_begin_local)
                else:
                    x_local -= learning_rate * (gradient)
    
            x_local_list.append(x_local)
    
        # Aggregate local solutions (simple averaging)
        x_global = np.mean(x_local_list, axis=0)
        # Compute norm of the difference between x_global and analytical solution
        norm_difference = np.linalg.norm(x_global - analytical_solution)**2
        norm_differences_FedHSA.append(norm_difference)
    
        # Check stopping criterion
        if norm_difference < threshold:
            print(f"Converged in {round + 1} rounds.")
            break
    results.append(norm_differences_FedHSA)

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

plt.savefig("comp_different_numbers.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()

