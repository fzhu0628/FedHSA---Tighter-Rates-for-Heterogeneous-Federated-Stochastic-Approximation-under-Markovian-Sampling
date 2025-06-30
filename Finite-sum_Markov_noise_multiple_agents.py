# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:14:34 2024

@author: 10123
"""

import numpy as np
import matplotlib.pyplot as plt

# def generate_symmetric_transition_matrix(n):
#     first_row = np.random.rand(1, n)
#     first_row /= sum(sum(first_row))
#     matrix = first_row
#     for i in range(n-1):
#         row_vec = np.zeros(n)
#         for j in range(i+1):
#             row_vec[j] = matrix[j][i+1]
#         row_vec[i+1:-1] = np.random.rand(n-i-1-1)
#         row_vec[-1] = np.random.rand(1) * (1 - np.sum(matrix[:, -1]))**2
#         row_vec[i+1:] = row_vec[i+1:] / sum(row_vec[i+1:]) * (1 - sum(row_vec[0:i+1]))
#         matrix = np.vstack((matrix, row_vec))
#     return matrix

def generate_symmetric_stochastic_matrix(n):
    """
    Generates a symmetric stochastic matrix of size n x n.
    
    Args:
        n (int): Size of the matrix (number of states).
    
    Returns:
        np.ndarray: Symmetric stochastic matrix.
    """
    # Step 1: Create a random symmetric matrix
    random_matrix = np.random.rand(n, n)
    symmetric_matrix = (random_matrix + random_matrix.T) / 2  # Ensure symmetry

    # Step 2: Normalize using the Sinkhorn-Knopp algorithm for doubly stochastic matrices
    for _ in range(100):  # Iterate to ensure convergence
        symmetric_matrix /= symmetric_matrix.sum(axis=0, keepdims=True)  # Normalize columns
        symmetric_matrix /= symmetric_matrix.sum(axis=1, keepdims=True)  # Normalize rows

    return symmetric_matrix
# Parameters
n_dim = 10       # Dimension of the optimization variable x
num_local_steps = 10  # Local steps before aggregation
heterogeneity_level = 1  # Controls how different the local loss functions are
n_samples = 10

threshold = 1e-20  # Convergence threshold
max_rounds = 4000  # Maximum number of rounds
learning_rate = 1e-3

target_smoothness = 10
target_convexity = 1

number_list = [1, 5, 20, 100]
results = []
for number in range(len(number_list)):
    num_agents = number_list[number]

    A_matrices_global = []
    b_vectors_global = []
    for m in range(num_agents):
        # Federated optimization routine with stopping criterion
        # Create Markov chains for gradient noise
        # Generate random local loss functions for agents
        A_matrices = []
        b_vectors = []
        
        
        for i in range(n_samples):
            # Generate symmetric matrix A_i and ensure positive definiteness after heterogeneity adjustment
            random_matrix = np.random.randn(n_dim, n_dim)
            A_i = np.dot(random_matrix.T, random_matrix) / n_dim  # Base symmetric matrix
            if i != 1:
                A_i += heterogeneity_level * np.random.randn(n_dim, n_dim) # No need to add that for single agent
            A_i = (A_i + A_i.T) / 2  # Ensure symmetry
            # A_i += np.eye(n_dim) * (np.max(-np.linalg.eigvalsh(A_i), 0) + 1e1) * num_agents
            A_i += np.eye(n_dim) * (np.max(-np.linalg.eigvalsh(A_i), 0) + 10)
            
            # Get eigenvalues of A_i
            eigenvalues = np.linalg.eigvalsh(A_i)
            lambda_min = np.min(eigenvalues)
            lambda_max = np.max(eigenvalues)
        
            # Scale and shift eigenvalues to fit [μ, L]
            A_i = ((target_smoothness - target_convexity) / (lambda_max - lambda_min)) * (A_i - lambda_min * np.eye(n_dim)) + target_convexity * np.eye(n_dim)
            A_matrices.append(A_i)
        
            # Generate random vector b_i
            b_i = np.random.randn(n_dim)
            if i != 1:
                b_i += heterogeneity_level * np.random.randn(n_dim)  # Adjust for heterogeneity
            b_vectors.append(b_i)
        A_matrices_global.append(A_matrices)
        b_vectors_global.append(b_vectors)
    # Compute the analytical solution to the global optimization problem
    A_global = np.zeros_like(A_matrices_global[0][0])
    b_global = np.zeros_like(b_vectors_global[0][0])
    for row in A_matrices_global:
        for matrix in row:
            A_global += matrix
    
    for row in b_vectors_global:
        for vector in row:
            b_global += vector
            
    analytical_solution = np.linalg.solve(A_global, b_global)
    
    # Initialize global variable x
    x_global_initial = np.random.randn(n_dim)
    
    
    norm_differences = []
    x_global = x_global_initial.copy()
    norm_differences.append(np.linalg.norm(x_global - analytical_solution)**2)
    
    P = []
    current_state = []
    
    for _ in range(num_agents):
        P.append(generate_symmetric_stochastic_matrix(n_samples)) # generate transition matrix with uniform stationary distribution)
        current_state.append(np.random.randint(0, n_samples))
    for _ in range(max_rounds):
        x_local_list = []
        gradient_begin_global = np.zeros_like(x_global)
        for m in range(num_agents):
            # gradient_begin_global += np.dot(A_matrices_global[m][current_state[m]], x_global) - b_vectors[current_state[m]]
            gradient_begin_global += np.dot(A_matrices_global[m][current_state[m]], x_global) - b_vectors_global[m][current_state[m]]
        gradient_begin_global /= num_agents
        for m in range(num_agents):
            x_local = x_global.copy()
    
            # Perform local steps using gradient descent with correction
            for step in range(num_local_steps):
                gradient = np.dot(A_matrices_global[m][current_state[m]], x_local) - b_vectors_global[m][current_state[m]] 
    
                next_state = np.random.choice(n_samples, p=P[m][current_state[m]])
                current_state[m] = next_state
                
                if step == 0:
                    gradient_begin_local = gradient.copy()
                
                x_local -= learning_rate * (gradient + gradient_begin_global - gradient_begin_local)
                # x_local -= learning_rate * gradient
    
    
            x_local_list.append(x_local)
    
        # Aggregate local solutions (simple averaging)
        x_global = np.mean(x_local_list, axis=0)
        # Compute norm of the difference between x_global and analytical solution
        norm_difference = np.linalg.norm(x_global - analytical_solution)**2
        norm_differences.append(norm_difference)
    
    results.append(norm_differences)

# #%%
# # Plot the convergence behavior
# plt.figure(figsize=(8, 5))
# plt.plot(results[0], label= str(number_list[0] ) + " agent")
# plt.plot(results[1], label= str(number_list[1] ) +" agents")
# plt.plot(results[2], label= str(number_list[2] ) +" agents")
# plt.plot(results[3], label= str(number_list[3] ) +" agents")
# plt.xlabel("Round")
# plt.ylabel(r'$||{\bar\theta}^{(t)}-\theta^\star||^2$')
# plt.title("Comparison between different number of agents")
# plt.yscale('log')
# plt.legend()
# plt.grid()
# plt.savefig("comp_finite_sum.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
# plt.show()

# # Plot the convergence behavior
# plt.figure(figsize=(8, 5))
# plt.semilogx(results[0], label= str(number_list[0] ) + " agent")
# plt.semilogx(results[1], label= str(number_list[1] ) +" agents")
# plt.semilogx(results[2], label= str(number_list[2] ) +" agents")
# plt.semilogx(results[3], label= str(number_list[3] ) +" agents")

# plt.xlabel("Round")
# plt.ylabel(r'$||{\bar\theta}^{(t)}-\theta^\star||^2$')
# plt.title("Comparison between different number of agents")
# plt.yscale('log')
# plt.legend()
# plt.grid()
# plt.show()
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

plt.savefig("comp_finite_sum.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()
