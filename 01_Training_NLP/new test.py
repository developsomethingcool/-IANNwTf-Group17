import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# Data Loading and Preprocessing
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
X = X / np.max(X)
y = digits.target.reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

# Data Visualization
def data_visualization_index(index):
    import matplotlib.pyplot as plt
    plt.imshow(digits.images[index], cmap='gray')
    plt.show()

def data_visualization():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(digits.images[i * 10 + j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()

# Data Generation and Minibatching
def data_generator(X, y, batch_size):
    while True:
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], y[batch_indices]

def simple_minibatch_generator(X, y, batch_size, num_epochs):
    num_batches = int(np.ceil(len(X) / batch_size))
    for epoch in range(num_epochs):
        generator = data_generator(X, y, batch_size)
        for batch in range(num_batches):
            yield next(generator)

# Activation Functions
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax:
    def __call__(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x):
        p = self.__call__(x)
        return p * (1 - p)

# MLP Layer Class
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation()

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.outputs = self.activation(self.z)

    def backward(self, d):
        d = d * self.activation.derivative(self.z)
        self.d_weights = np.dot(self.inputs.T, d)
        self.d_biases = np.sum(d, axis=0, keepdims=True)
        self.d_inputs = np.dot(d, self.weights.T)
        return self.d_inputs

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

# Loss Function
class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(y_pred + 1e-15))
        return loss / len(y_pred)

    def derivative(self):
        return self.y_pred - self.y_true

# MLP Backward Pass
def MLP_backward(model, loss_derivative):
    for i in reversed(range(len(model))):
        loss_derivative = model[i].backward(loss_derivative)
    return loss_derivative

# Training the MLP
def train_mlp(X_train, y_train, X_val, y_val, layer_sizes, activations, loss, learning_rate, batch_size, num_epochs):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))
    loss_fn = loss()
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        for X_batch, y_batch in simple_minibatch_generator(X_train, y_train, batch_size, num_epochs):
            for layer in layers:
                layer.forward(X_batch)
                X_batch = layer.outputs
            loss = loss_fn(layer.outputs, y_batch)
            train_loss += loss
            loss_derivative = loss_fn.derivative()
            MLP_backward(layers, loss_derivative)
            for layer in layers:
                layer.update(learning_rate)
        for X_batch, y_batch in simple_minibatch_generator(X_val, y_val, batch_size, num_epochs):
            for layer in layers:
                layer.forward(X_batch)
                X_batch = layer.outputs
            loss = loss_fn(layer.outputs, y_batch)
            val_loss += loss
        train_loss_history.append(train_loss / len(X_train))
        val_loss_history.append(val_loss / len(X_val))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_history[-1]:.4f}, Val Loss: {val_loss_history[-1]:.4f}")
    return layers, train_loss_history, val_loss_history

# Model Testing and Evaluation
def forward_pass(X, layers):
    for layer in layers:
        layer.forward(X)
        X = layer.outputs
    return X

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# Main Execution
layer_sizes = [64, 32, 10]
activations = [Sigmoid, Sigmoid, Softmax]
loss = CrossEntropyLoss
learning_rate = 0.1
batch_size = 32
num_epochs = 50

num_train = int(0.8 * len(X))
X_train, y_train = X[:num_train], y[:num_train]
X_val, y_val = X[num_train:], y[num_train:]

layers, train_loss_history, val_loss_history = train_mlp(X_train, y_train, X_val, y_val, layer_sizes, activations, loss, learning_rate, batch_size, num_epochs)

y_pred = forward_pass(X_val, layers)
acc = accuracy(y_pred, y_val)
print(f"Validation Accuracy: {acc:.4f}")
