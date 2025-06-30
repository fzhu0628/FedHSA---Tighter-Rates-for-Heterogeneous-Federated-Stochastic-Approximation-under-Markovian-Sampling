# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:14:34 2024

@author: 10123
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_agents = 20  # Number of agents
n_dim = 10       # Dimension of the optimization variable x
num_local_steps = 10  # Local steps before aggregation
heterogeneity_level = 100  # Controls how different the local loss functions are
noise_variance = 5e-1  # Variance of the gradient noise
# Initialize global variable x
x_global_initial = np.random.randn(n_dim)
threshold = 1e-20  # Convergence threshold
max_rounds = 2000  # Maximum number of rounds
learning_rate = 1e-2

# Generate random local loss functions for agents
A_matrices = []
b_vectors = []

for i in range(num_agents):
    # Generate symmetric matrix A_i and ensure positive definiteness after heterogeneity adjustment
    random_matrix = np.random.randn(n_dim, n_dim)
    A_i = np.dot(random_matrix.T, random_matrix) / n_dim  # Base symmetric matrix
    # A_i += heterogeneity_level * np.random.randn(n_dim, n_dim)
    A_i = (A_i + A_i.T) / 2  # Ensure symmetry
    A_i += np.eye(n_dim) * (np.max(-np.linalg.eigvalsh(A_i), 0) + 1 ) # Ensure positive definiteness
    A_matrices.append(A_i)

    # Generate random vector b_i
    b_i = np.random.randn(n_dim)
    b_i += heterogeneity_level * np.random.randn(n_dim)  # Adjust for heterogeneity
    b_vectors.append(b_i)

# Compute the analytical solution to the global optimization problem
A_global = sum(A_matrices)
b_global = sum(b_vectors)
analytical_solution = np.linalg.solve(A_global, b_global)


#%% Vanilla SGD method
# Federated optimization routine with stopping criterion

# Create Markov chains for gradient noise
markov_states = []
transition_matrices = []

for i in range(num_agents):
    # Create a transition matrix with rows summing to 1 and non-negative entries
    vector = np.random.rand(n_dim)  # Ensure non-negativity
    markov_states.append(vector)
    
    matrix = np.random.randn(n_dim, n_dim)
    matrix = matrix / np.max(np.abs(np.linalg.eigvals(matrix))) / 1.2  # Normalize spectral radius to < 1
    transition_matrices.append(matrix)

norm_differences_SGD = []
x_global = x_global_initial.copy()
norm_differences_SGD.append(np.linalg.norm(x_global - analytical_solution)**2)
for round in range(max_rounds):
    x_local_list = []

    for i in range(num_agents):
        x_local = x_global.copy()

        # Perform local steps using gradient descent
        for step in range(num_local_steps):
            gradient = np.dot(A_matrices[i], x_local) - b_vectors[i]
            
            # Add Markovian noise to the gradient
            noise = markov_states[i]
            gradient += noise
            
            # Update Markov state
            markov_states[i] = np.dot(transition_matrices[i], markov_states[i]) + noise_variance * np.random.randn(n_dim)
            

            x_local -= learning_rate * gradient

        x_local_list.append(x_local)

    # Aggregate local solutions (simple averaging)
    x_global = np.mean(x_local_list, axis=0)

    # Compute norm of the difference between x_global and analytical solution
    norm_difference = np.linalg.norm(x_global - analytical_solution)**2
    norm_differences_SGD.append(norm_difference)

    # Check stopping criterion
    if norm_difference < threshold:
        print(f"Converged in {round + 1} rounds.")
        break
#%% FedHSA 
# Federated optimization routine with stopping criterion
# Create Markov chains for gradient noise
markov_states = []
transition_matrices = []

for i in range(num_agents):
    # Create a transition matrix with rows summing to 1 and non-negative entries
    vector = np.random.rand(n_dim)  # Ensure non-negativity
    markov_states.append(vector)
    
    matrix = np.random.randn(n_dim, n_dim)
    matrix = matrix / np.max(np.abs(np.linalg.eigvals(matrix))) / 1.2  # Normalize spectral radius to < 1
    transition_matrices.append(matrix)

norm_differences_FedHSA = []
x_global = x_global_initial.copy()
norm_differences_FedHSA.append(np.linalg.norm(x_global - analytical_solution)**2)
for round in range(max_rounds):
    x_local_list = []
    gradient_begin_global = np.dot(np.mean(A_matrices, axis=0), x_global) - np.mean(b_vectors, axis=0) + np.mean(markov_states, axis=0)
    for i in range(num_agents):
        x_local = x_global.copy()

        # Perform local steps using gradient descent with correction
        for step in range(num_local_steps):
            gradient = np.dot(A_matrices[i], x_local) - b_vectors[i]
            
            # Add Markovian noise to the gradient
            noise = markov_states[i]
            gradient += noise

            # Update Markov state
            markov_states[i] = np.dot(transition_matrices[i], markov_states[i]) + noise_variance * np.random.randn(n_dim)
            
            if step == 0:
                gradient_begin_local = gradient
            
            x_local -= learning_rate * (gradient + gradient_begin_global - gradient_begin_local)

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


#%%
# Plot the convergence behavior
plt.figure(figsize=(8, 5))
plt.plot(norm_differences_SGD,'--o', color='b', linewidth=2, markevery=50, label="Local SA")
plt.plot(norm_differences_FedHSA,'-^', color='r', linewidth=2,markevery=50, label="FedHSA")
plt.xlabel(r'${{t}}$', fontsize=30, fontname='Times New Roman', weight='bold')
plt.ylabel(r'$E_t$', fontsize=30, fontname='Times New Roman', weight='bold')
plt.yscale('log')
plt.legend(loc='best', fontsize=20)
# Grid
plt.grid(True, which="both", linestyle="--", linewidth=0.7)
# Axes properties
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

plt.savefig("comp_vanilla_noisy.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()
