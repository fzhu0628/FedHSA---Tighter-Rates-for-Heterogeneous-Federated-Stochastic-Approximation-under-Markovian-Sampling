import numpy as np
import matplotlib.pyplot as plt
import copy

# Parameters
n_dim = 10       # Dimension of the optimization variable x
num_local_steps = 1  # Local steps before aggregation
noise_variance = 1e-1  # Variance of the gradient noise
total_samples = 100  # Fixed total number of samples across all agents

# Generate synthetic data for the entire dataset
np.random.seed(42)  # For reproducibility
X = np.random.randn(total_samples, n_dim)
y = np.random.randn(total_samples)



# Compute the analytical solution to the global least squares problem

analytical_solution = np.linalg.solve(X.T @ X, X.T @ y)



# Initialize global variable x
x_global_initial = np.random.randn(n_dim)

# Federated optimization routine with stopping criterion
threshold = 1e-20  # Convergence threshold
max_iterations = 10000  # Maximum number of iterations
learning_rate = 1e-3

number_list = [1, 4, 10, 20]
results = []
for number in range(4):
    num_agents = number_list[number]
    
    
    # Create Markov chains for gradient noise
    # markov_states = []
    # transition_matrices = []

    # for i in range(num_agents):
    #     # Create a transition matrix with rows summing to 1 and non-negative entries
    #     vector = np.random.rand(n_dim)  # Ensure non-negativity
    #     markov_states.append(vector)
        
    #     matrix = np.random.randn(n_dim, n_dim)
    #     matrix = matrix / np.max(np.abs(np.linalg.eigvals(matrix)))  # Normalize spectral radius to < 1
    #     transition_matrices.append(matrix)

    # Partition data among agents
    X_agents = np.array_split(X, num_agents)
    y_agents = np.array_split(y, num_agents)

    norm_differences = []
    x_global = copy.deepcopy(x_global_initial)
    for iteration in range(max_iterations):
        # Create iid noise
        iid_noise = []
        
        for i in range(num_agents):
            iid_noise.append(np.random.normal(0, noise_variance, size=x_global.shape))
        
        # Compute norm of the difference between x_global and analytical solution
        norm_difference = np.linalg.norm(x_global - analytical_solution)**2
        norm_differences.append(norm_difference)
        # x_local_list = []
        gradient = 0
        for i in range(num_agents):
            # x_local = copy.deepcopy(x_global)
            # gradient_begin_global = 0
            # for k in range(num_agents):
            #     gradient_begin_global += X_agents[k].T @ (X_agents[k] @ x_global - y_agents[k])
            # gradient_begin_global /= num_agents
            # gradient_begin_global += np.mean(markov_states, axis=0)
            # gradient_begin_global += np.mean(iid_noise, axis=0)
            
            
            # Perform local steps using gradient descent
            for step in range(num_local_steps):
                # gradient = 2 * X_agents[i].T @ (X_agents[i] @ x_local - y_agents[i])
                local_gradient = 2 * X_agents[i].T @ (X_agents[i] @ x_global - y_agents[i])

                # Add Markovian noise to the gradient
                # noise = markov_states[i]
                noise = iid_noise[i]
                
                gradient += local_gradient / num_agents

                # Update Markov state
                # markov_states[i] = np.dot(transition_matrices[i], markov_states[i]) + noise_variance * np.random.randn(n_dim)

                # if step == 0:
                #     gradient_begin_local = copy.deepcopy(gradient)
                
                # learning_rate = 0.001
                # x_local -= learning_rate * (gradient + gradient_begin_global - gradient_begin_local) * num_agents
                # x_local -= learning_rate * gradient * num_agents

            # x_local_list.append(x_local)

        # Aggregate local solutions (simple averaging)
        # x_global = np.mean(x_local_list, axis=0)
        x_global -= learning_rate * gradient

        

        # Check stopping criterion
        if norm_difference < threshold:
            print(f"Converged in {iteration + 1} iterations.")
            break
    results.append(norm_differences)

#%%
# Plot the convergence behavior
plt.figure(figsize=(8, 5))
plt.semilogx(results[0], label="1 agent")
plt.semilogx(results[1], label="4 agents")
plt.semilogx(results[2], label="10 agents")
plt.semilogx(results[3], label="20 agents")
plt.xlabel("Iteration")
plt.ylabel("Norm of Difference")
plt.title("Convergence of Federated Least Squares Optimization")
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# Print the final global solution
print("Final global solution:", x_global)
print("Analytical solution:", analytical_solution)
