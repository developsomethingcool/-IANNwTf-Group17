
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# Data Loading and Preprocessing
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    X = X / 16
    return X, y

# Data Visualization
def data_visualization_index(X, y, index):
    print("Label: ", y[index])
    print("Data: ")
    for i in range(8):
        for j in range(8):
            print(int(X[index][i * 8 + j] * 16), end=" ")
        print()

def data_visualization(X, y):
    print("Label: ", y)
    print("Data: ")
    for i in range(8):
        for j in range(8):
            print(int(X[i * 8 + j] * 16), end=" ")
        print()

# Data Generation and Minibatching
def data_generator(X, y, batch_size):
    while True:
        indices = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[indices[i:i+batch_size]]
            y_batch = y[indices[i:i+batch_size]]
            yield X_batch, y_batch

def simple_minibatch_generator(X, y, batch_size):
    data_gen = data_generator(X, y, batch_size)
    while True:
        X_batch, y_batch = next(data_gen)
        yield X_batch, y_batch

# Activation Functions
class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def backward(a):
        return a * (1 - a)

class Softmax:
    @staticmethod
    def forward(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def backward(a, y):
        return a - y

# MLP Layer Class
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        self.A = self.activation.forward(self.Z)
        return self.A

    def backward(self, dA):
        dZ = dA * self.activation.backward(self.A)
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        return dX

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Loss Function
class CrossEntropyLoss:
    @staticmethod
    def forward(y_pred, y_true):
        loss = -np.sum(y_true * np.log(y_pred + 1e-10))
        return loss / y_pred.shape[0]

    @staticmethod
    def backward(y_pred, y_true):
        return y_pred - y_true

# MLP Backward Pass
def MLP_backward(model, loss_grad):
    for i in reversed(range(len(model))):
        loss_grad = model[i].backward(loss_grad)
    return loss_grad

# Training the MLP
def train_mlp(X_train, y_train, X_val, y_val, layer_sizes, activations, learning_rate, epochs, batch_size):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    train_gen = simple_minibatch_generator(X_train, y_train, batch_size)
    val_gen = simple_minibatch_generator(X_val, y_val, batch_size)

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        # Training
        for X_batch, y_batch in train_gen:
            A = X_batch
            for layer in layers:
                A = layer.forward(A)
            loss = CrossEntropyLoss.forward(A, y_batch)
            train_loss += loss
            loss_grad = CrossEntropyLoss.backward(A, y_batch)
            MLP_backward(layers, loss_grad)
            for layer in layers:
                layer.update(learning_rate)

            y_pred = np.argmax(A, axis=1)
            y_true = np.argmax(y_batch, axis=1)
            train_acc += np.sum(y_pred == y_true) / y_true.shape[0]

        # Validation
        for X_batch, y_batch in val_gen:
            A = X_batch
            for layer in layers:
                A = layer.forward(A)
            loss = CrossEntropyLoss.forward(A, y_batch)
            val_loss += loss

            y_pred = np.argmax(A, axis=1)
            y_true = np.argmax(y_batch, axis=1)
            val_acc += np.sum(y_pred == y_true) / y_true.shape[0]

        train_loss /= X_train.shape[0] / batch_size
        val_loss /= X_val.shape[0] / batch_size
        train_acc /= X_train.shape[0] / batch_size
        val_acc /= X_val.shape[0] / batch_size

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return layers

# Model Testing and Evaluation
def forward_pass(X, layers):
    A = X
    for layer in layers:
        A = layer.forward(A)
    return A

def test_mlp(X_test, y_test, layers):
    y_pred = forward_pass(X_test, layers)
    loss = CrossEntropyLoss.forward(y_pred, y_test)
    acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")

# Main Execution
if __name__ == "__main__":
    X, y = load_data()
    X_train, y_train = X[:1200], y[:1200]
    X_val, y_val = X[1200:1350], y[1200:1350]
    X_test, y_test = X[1350:], y[1350:]

    layer_sizes = [64, 32, 10]
    activations = [Sigmoid(), Sigmoid(), Softmax()]
    learning_rate = 0.1
    epochs = 100
    batch_size = 32

    layers = train_mlp(X_train, y_train, X_val, y_val, layer_sizes, activations, learning_rate, epochs, batch_size)
    test_mlp(X_test, y_test, layers)
