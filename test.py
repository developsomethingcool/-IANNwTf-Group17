import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation_functions):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_functions = activation_functions
        
        # Initialize layers
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layer = MLP_Layer(sizes[i+1], activation_functions[i])
            layer.init_weights(sizes[i])
            self.layers.append(layer)
    
    def forward(self, x):
        # Compute the forward pass through the network
        a = x
        for layer in self.layers:
            a = layer.Neuron_activation(a)
        return a
    
    def backward(self, x, y, learning_rate):
        # Compute the backward pass through the network and update weights and biases
        grad_output = self.loss.backward(self.predict(x), y)
        for layer in reversed(self.layers):
            grad_input, grad_weights, grad_bias = layer.backward(grad_output)
            layer.weights -= learning_rate * grad_weights
            layer.bias -= learning_rate * grad_bias
            grad_output = grad_input
    
    def train(self, X, y, learning_rate, epochs):
        # Train the network on the given data for the specified number of epochs
        for i in range(epochs):
            for j in range(len(X)):
                x = X[j]
                y_true = y[j]
                self.backward(x, y_true, learning_rate)
    
    def predict(self, X):
        # Make predictions on the given input data
        y_pred = []
        for x in X:
            y = self.forward(x)
            y_pred.append(y)
        return y_pred

class MLP_Layer:
    def __init__(self, n_neurons, activation_function):
        self.n_neurons = n_neurons
        self.activation_function = activation_function
        self.weights = None
        self.bias = None
        self.activation = None
        self.preactivation = None

    def init_weights(self, n_inputs):
        self.weights = np.random.randn(self.n_neurons, n_inputs)
        self.bias = np.zeros((self.n_neurons, 1))

    def Neuron_activation(self, inputs):
        # Compute the activation of the layer's neurons.
        try:
            self.preactivation = np.dot(self.weights, inputs) + self.bias.ravel()
        except:
            self.preactivation = np.dot(self.weights, inputs) + self.bias
        self.activation = self.activation_function.call(self.preactivation)
        return self.activation

    def backward(self, grad_output):
        grad_activation = self.activation_function.gradient(self.preactivation) * grad_output
        grad_weights = np.dot(grad_activation, self.weights)
        grad_bias = np.sum(grad_activation, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_activation)
        return grad_input, grad_weights, grad_bias

class Sigmoid:
    def __init__(self):
        self.result = None
        pass

    def call(self, z):
        result = expit(z)
        self.result = result    
        return result

    def gradient(self,y):
        # Calculate the gradient of the sigmoid function for a single value or array
        sigmoid_x = self.result
        gradient_x = sigmoid_x * (1 - sigmoid_x)

        # Return the gradient
        return gradient_x

class Softmax:
    def __init__(self):
        self.result = None  # Stores the result of the softmax function
        pass

    def call(self, x):
        # Calculate the softmax activation for a given array 'x'
        try:
             exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtracting the maximum value for numerical stability
        except:
            exp_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
       
        self.result = exp_x / exp_x.sum(axis=-1, keepdims=True)


        return self.result

    def gradient(self, y):
        # The gradient of the softmax function is the same as the softmax probabilities
        gradient = self.result - y
        return gradient

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None  # Stores the predicted probabilities

    def forward(self, probs, targets):
        # Calculate the cross-entropy loss for a batch of predictions and targets

        # Clip the predicted probabilities to prevent numerical instability
        epsilon = 1e-15  # Small constant to avoid taking the log of zero
        probs = np.clip(probs, epsilon, 1 - epsilon)

        # Manually calculate cross-entropy and return an array of losses for each sample in the batch
        losses = -np.sum(targets * np.log(probs), axis=1)
        mean_loss = np.mean(losses)
        return mean_loss

    def backward(self, probs, targets):
        # Calculate the gradient of the cross-entropy loss
        d_loss = probs - targets
        return d_loss

np.random.seed(42)
X_train = np.random.rand(100, 5)  # 100 samples with 5 features each
y_train = np.random.randint(0, 2, size=(100, 3))  # 3 classes, one-hot encoded labels

# Define MLP parameters
input_size = 5
hidden_sizes = [10, 5]
output_size = 3
activation_functions = [Sigmoid(), Sigmoid(), Softmax()]

# Initialize MLP model
mlp_model = MLP(input_size, hidden_sizes, output_size, activation_functions)

# Define training parameters
learning_rate = 0.01
epochs = 1000

# Train the model
mlp_model.train(X_train, y_train, learning_rate, epochs)