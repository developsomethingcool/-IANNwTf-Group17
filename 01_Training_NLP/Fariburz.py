import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the digits dataset and preprocess it
digits = load_digits()
X = digits.data
y = digits.target
#X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=True)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Initialize the weights and biases for the MLP
n_input = X_train.shape[1]
n_hidden = 16
n_output = y_train.shape[1]
W1 = np.random.randn(n_input, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output)
b2 = np.zeros((1, n_output))

# Define the activation function and its derivative
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
    

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the forward propagation function
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)
    return y_hat, (z1, a1, z2)

# Define the backward propagation function
def backward_propagation(X, y, y_hat, cache):
    z1, a1, z2 = cache
    delta2 = (y_hat - y) * sigmoid_derivative(z2)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(z1)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0)
    return dW1, db1, dW2, db2

# Define the update function to update the weights and biases
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Train the MLP using the training set
learning_rate = 0.1
n_iterations = 1000
for i in range(n_iterations):
    # Forward propagation
    y_hat, cache = forward_propagation(X_train, W1, b1, W2, b2)
    # Backward propagation
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, y_hat, cache)
    # Update parameters
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

# Test the MLP using the testing set
y_hat_test, _ = forward_propagation(X_test, W1, b1, W2, b2)

# Evaluate the performance of the MLP
accuracy = np.mean(np.argmax(y_hat_test, axis=1) == np.argmax(y_test, axis=1))
print("Accuracy:", accuracy)
