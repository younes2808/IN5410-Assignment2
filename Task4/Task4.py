import numpy as np

print("TASK 4")
print("-" * 60)

# Parameters
# Input data
x1 = 0.04
x2 = 0.20

# Target output
y_target = 0.50

# Learning rate
alpha = 0.4

# Helper functions
# Activation function: Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Training process
# Reset weights to initial values
w1, w2, w3, w4, w5, w6 = np.random.rand(6)

threshold = 1e-6
max_iterations = 100000
errors = []

print(f"\nTraining parameters:")
print(f"  Learning rate: {alpha}")
print(f"  Threshold: {threshold}")
print(f"  Max iterations: {max_iterations}")

previous_error = float('inf')

for iteration in range(max_iterations):
    # FORWARD PROPAGATION
    z_h1 = w1 * x1 + w2 * x2
    h1 = sigmoid(z_h1)
    
    z_h2 = w3 * x1 + w4 * x2
    h2 = sigmoid(z_h2)
    
    z_out = w5 * h1 + w6 * h2
    y_pred = sigmoid(z_out)
    
    # Calculate error
    current_error = 0.5 * (y_target - y_pred) ** 2
    #Keep track of errors
    errors.append(current_error)
    
    # Print first 10 rounds
    if iteration < 10:
        print(f"Round {iteration + 1}: Error = {current_error:.8f}")
    
    # Check stopping condition
    error_diff = abs(previous_error - current_error)
    if error_diff < threshold:
        print(f"\nRound {iteration + 1}: Error = {current_error:.8f}")
        print(f"\nTraining stopped at iteration {iteration + 1}")
        print(f"Error difference: {error_diff:.10f} < threshold {threshold}")
        break
    
    # BACKPROPAGATION
    delta_out = (y_pred - y_target) * sigmoid_derivative(z_out)
    
    grad_w5 = delta_out * h1
    grad_w6 = delta_out * h2
    
    delta_h1 = delta_out * w5 * sigmoid_derivative(z_h1)
    delta_h2 = delta_out * w6 * sigmoid_derivative(z_h2)
    
    grad_w1 = delta_h1 * x1
    grad_w2 = delta_h1 * x2
    grad_w3 = delta_h2 * x1
    grad_w4 = delta_h2 * x2
    
    # UPDATE WEIGHTS
    w1 = w1 - alpha * grad_w1
    w2 = w2 - alpha * grad_w2
    w3 = w3 - alpha * grad_w3
    w4 = w4 - alpha * grad_w4
    w5 = w5 - alpha * grad_w5
    w6 = w6 - alpha * grad_w6
    
    previous_error = current_error

print(f"\n--- FINAL RESULTS ---")
print(f"Final weights:")
print(f"  w1 = {w1:.6f}")
print(f"  w2 = {w2:.6f}")
print(f"  w3 = {w3:.6f}")
print(f"  w4 = {w4:.6f}")
print(f"  w5 = {w5:.6f}")
print(f"  w6 = {w6:.6f}")
print(f"Final prediction: {y_pred:.6f}")
print(f"Target: {y_target}")
print(f"Final error: {current_error:.10f}")

# Save error table for first 10 and last round
print("ERROR TABLE")
print(f"{'Round':<10} {'Error':<20}")
for i in range(min(10, len(errors))):
    print(f"{i+1:<10} {errors[i]:<20.10f}")
if len(errors) > 10:
    print(f"{len(errors):<10} {errors[-1]:<20.10f}")