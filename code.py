import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

independent_file_path = 'data/independent.csv'
dependent_file_path = 'data/dependent.csv'

independent_data = pd.read_csv(independent_file_path)
dependent_data = pd.read_csv(dependent_file_path)

# Combine independent and dependent datasets
data = independent_data.copy()
data['Target'] = dependent_data.values

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term to X
X_bias = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# Perform batch gradient descent for logistic regression
def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Initialize parameters
theta = np.zeros(X_bias.shape[1])
learning_rate = 0.1
num_iterations = 1000

# Train the model
theta_final, cost_history = gradient_descent(X_bias, y, theta, learning_rate, num_iterations)

# Final cost function value
final_cost = compute_cost(X_bias, y, theta_final)

# Print final coefficients and cost
print("Final Coefficients (Theta):", theta_final)
print("Final Cost Function Value:", final_cost)

# Plot the cost function vs. iterations (First 50 iterations)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 51), cost_history[:50], marker='o', label="Cost")
plt.title("Cost Function vs. Iterations (First 50)")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.grid()
plt.legend()
plt.show()

# Plot the dataset with the decision boundary
plt.figure(figsize=(8, 6))

# Plot data points
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_scaled[y == label, 0], X_scaled[y == label, 1], label=f"Class {label}")

# Plot decision boundary
x_values = np.linspace(-2, 2, 100)
y_values = -(theta_final[0] + theta_final[1] * x_values) / theta_final[2]
plt.plot(x_values, y_values, color='red', label="Decision Boundary")

plt.title("Dataset with Decision Boundary")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.grid()
plt.show()

# Add new features (squares of original features) and train again
X_extended = np.hstack((X_scaled, X_scaled[:, 0:1]**2, X_scaled[:, 1:2]**2))
X_extended_bias = np.hstack((np.ones((X_extended.shape[0], 1)), X_extended))

# Train logistic regression with extended dataset
theta_extended = np.zeros(X_extended_bias.shape[1])
theta_extended_final, cost_extended = gradient_descent(X_extended_bias, y, theta_extended, learning_rate, num_iterations)

# Plot the extended dataset with decision boundary
plt.figure(figsize=(8, 6))

# Plot data points
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_scaled[y == label, 0], X_scaled[y == label, 1], label=f"Class {label}")

# Plot quadratic decision boundary
def quadratic_boundary(x1, x2, theta):
    return theta[0] + theta[1] * x1 + theta[2] * x2 + theta[3] * x1**2 + theta[4] * x2**2

x1_range = np.linspace(-2, 2, 100)
x2_range = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = quadratic_boundary(X1, X2, theta_extended_final)
plt.contour(X1, X2, Z, levels=[0], colors="red", label="Decision Boundary")

plt.title("Extended Dataset with Quadratic Decision Boundary")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix and Metrics
y_pred = (sigmoid(X_bias.dot(theta_final)) >= 0.5).astype(int)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
