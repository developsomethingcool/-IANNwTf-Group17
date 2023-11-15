
import numpy as np
from sklearn.datasets import load_digits

# 2.1 Load data
digits = load_digits()
X = digits.data
y = digits.target

# 2.2 Sigmoid activation function
class Sigmoid:
    def __call__(self, x):
        clipped_x = np.clip(x, -500, 500)  # limit the range of x to prevent overflow
        self.output = 1 / (1 + np.exp(-clipped_x))
        return self.output

    def backward(self, dA):
        return dA * self.output * (1 - self.output)

# 2.3 Softmax activation function
class Softmax:
    def __call__(self, x):
        self.exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return self.exp_x / np.sum(self.exp_x, axis=1, keepdims=True)

    def backward(self, dA):
        dZ = np.zeros_like(dA)
        for i in range(dA.shape[0]):
            jacobian_matrix = np.diag(self.exp_x[i]) - np.outer(self.exp_x[i], self.exp_x[i])
            dZ[i] = np.dot(jacobian_matrix, dA[i])
        return dZ

# 2.4 MLP layer class
class MLP_Layer:
    def __init__(self, activation, n_units, input_size):
        self.activation = activation
        self.n_units = n_units
        self.input_size = input_size
        self.weights = np.random.randn(input_size, n_units) * 0.01
        self.bias = np.zeros((1, n_units))

    def forward(self, input):
        
        self.input = input
        self.z = np.dot(input, self.weights) + self.bias
        if self.activation == "sigmoid":
            sigmoid_activation = Sigmoid()
            return sigmoid_activation(self.z)
        elif self.activation == "softmax":
            softmax_activation = Softmax()
            return softmax_activation(self.z)

    def backward(self, dA):
        if self.activation == "sigmoid":
            sig = Sigmoid()(self.z)
            dZ = dA * sig * (1 - sig)
        elif self.activation == "softmax":
            dZ = dA
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev, dW, db

# 2.5 MLP class
class MLP:
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layer = MLP_Layer(activations[i], layer_sizes[i], X.shape[1])
            else:
                layer = MLP_Layer(activations[i], layer_sizes[i], layer_sizes[i-1])
            self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output, y):
        dA = self.loss.backward(output, y)
        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA)
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db

    def train(self, X, y, minibatch_size, n_epochs, learning_rate, loss):
        self.learning_rate = learning_rate
        self.loss = loss
        for epoch in range(n_epochs):
            for i in range(0, X.shape[0], minibatch_size):
                X_batch = X[i:i+minibatch_size]
                y_batch = y[i:i+minibatch_size]
                output = self.forward(X_batch)
                self.backward(output, y_batch)

# 2.6 CCE loss function
class CCE:
    def __call__(self, output, y):
        m = y.shape[0]
        sigmoid = Sigmoid()
        p = sigmoid(output)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss
    def backward(self, output, y):
        m = y.shape[0]
        grad = Softmax()(output)
        grad[range(m), y] -= 1
        grad /= m
        return grad

# 3.1 CCE Backwards
def CCE_backwards(output, y):
    m = y.shape[0]
    grad = softmax(output)
    grad[range(m), y] -= 1
    grad /= m
    return grad



# 3.3 MLP Layer Weights Backwards
def MLP_Layer_weights_backwards(dA, input):
    dW = np.dot(input.T, dA)
    db = np.sum(dA, axis=0, keepdims=True)
    dA_prev = np.dot(dA, input.T)
    return dA_prev, dW, db

# 3.4 MLP Layer Backwards
def MLP_Layer_backwards(dA, layer, input):
    dZ = dA
    dA_prev, dW, db = layer.backward(dZ)
    dA_prev, dW, db = MLP_Layer_weights_backwards(dA_prev, input)
    return dA_prev, dW, db

# 3.5 Gradient Tape and MLP Backward
def MLP_Backward(X, y, layers, loss, learning_rate):
    output = X
    for layer in layers:
        output = layer.forward(output)
    loss_value = loss(output, y)
    dA = loss.backward(output, y)
    for layer in reversed(layers):
        dA, dW, db = layer.backward(dA)
        layer.weights -= learning_rate * dW
        layer.bias -= learning_rate * db
    return loss_value

# 3.6 Training
class MLP:
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layer = MLP_Layer(activations[i], layer_sizes[i], X.shape[1])
            else:
                layer = MLP_Layer(activations[i], layer_sizes[i], layer_sizes[i-1])
            self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output, y):
        dA = self.loss.backward(output, y)
        for layer in reversed(self.layers):
            dA, dW, db = layer.backward(dA)
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db

    def train(self, X, y, minibatch_size, n_epochs, learning_rate, loss):
        self.learning_rate = learning_rate
        self.loss = loss
        for epoch in range(n_epochs):
            for i in range(0, X.shape[0], minibatch_size):
                X_batch = X[i:i+minibatch_size]
                y_batch = y[i:i+minibatch_size]
                output = self.forward(X_batch)
                self.backward(output, y_batch)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
mlp = MLP(layer_sizes=[10, 5], activations=['relu', 'softmax'])
mlp.train(X, y, minibatch_size=32, n_epochs=10, learning_rate=0.01, loss=CCE())
y_pred = mlp.predict(X)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)
