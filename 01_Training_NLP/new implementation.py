import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the digits dataset
data = load_digits()
inputs = data.data
targets = data.target

# Function to visualize a specific data point at a given index
def data_visualization_index(data, target, index):
    # Create a new figure for the plot with an 8x8-inch size
    plt.figure(figsize=(8, 8))
    
    # Display the image of the data point at the specified index, reshaped to 8x8, in grayscale
    plt.imshow(data[index].reshape(8, 8), cmap='gray')
    
    # Set the title of the plot to the corresponding target value
    plt.title(target[index])
    
    # Turn off the axis labels and ticks for a cleaner visualization
    plt.axis('off')
    
    # Show the plot on the screen
    plt.show() 

# Function to visualize the entire dataset as a single image
def data_visualization(data, target):
    # Create a new figure for the plot with an 8x8-inch size
    plt.figure(figsize=(8, 8))
    
    # Display the entire dataset as a single image, reshaped to 8x8, in grayscale
    plt.imshow(data.reshape(8, 8), cmap='gray')
    
    # Set the title of the plot to the target label (assuming it's a single label for the entire dataset)
    plt.title(target)
    
    # Turn off the axis labels and ticks for a cleaner visualization
    plt.axis('off')
    
    # Show the plot on the screen
    plt.show()

# Step 3: Reshape the 8x8 images into vectors
inputs = np.reshape(inputs, (len(inputs), -1))

# Step 4: Rescale the images to the [0, 1] range
inputs = inputs.astype('float32') / 16.0

# Step 5: One-hot encode the target digits
one_hot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
one_hot_targets = one_hot_encoder.fit_transform(targets.reshape(-1, 1))

# Step 6: Write a data generator function to shuffle and yield mini-batches
def data_generator(input_data, target_data, minibatch_size):
    # Get the total number of samples in the dataset
    num_samples = len(input_data)
    
    # Create an array of indices corresponding to the samples
    indices = np.arange(num_samples)
    # Shuffle the indices to randomize the order of samples
    np.random.shuffle(indices)

    # Iterate over the shuffled indices to yield mini-batches
    for start_idx in range(0, num_samples - minibatch_size + 1, minibatch_size):
        # Select a subset of indices for the current mini-batch
        excerpt = indices[start_idx:start_idx + minibatch_size]
        
        # Yield the corresponding input and target data for the mini-batch
        yield input_data[excerpt], target_data[excerpt]

# Step 7: Adjust the generator function to create minibatches
def simple_minibatch_generator(inputs, targets, batch_size):
    generator = data_generator(inputs, targets, batch_size)
    for inputs_batch, targets_batch in generator:
        yield inputs_batch, targets_batch

# Split the data into training and testing sets
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, one_hot_targets, test_size=0.2, random_state=42)

class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, z):
        # Calculate the sigmoid activation for the input z
        return 1. / (1. + np.exp(-z))

    def backward(self, pre_activation, activation, d_activation):
        # Compute the derivative of the pre-activation with respect to the activation
        d_pre_activation = activation * (1 - activation) * d_activation
        return d_pre_activation


class Softmax:
    def __init__(self):
        pass

    def __call__(self, z):
        # Ensure numerical stability
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def backward(self, pre_activation, activation, d_activation):
        # Compute the derivative of the pre-activation with respect to the activation
        d_pre_activation = d_activation * activation * (1 - activation)
        return d_pre_activation



class Layer:
    def __init__(self, input_size, num_units, activation):
        self.input_size = input_size
        self.num_units = num_units
        self.activation = activation

        # Initialize weights and biases
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(input_size, num_units))
        self.biases = np.zeros(num_units)

    def forward(self, inputs):
        # Compute pre-activations
        pre_activations = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        return self.activation(pre_activations)

    def backward_weights(self, d_pre_activations, inputs):
        # Compute the gradient of the loss with respect to the weights
        dW = np.dot(inputs.T, d_pre_activations)
        # Compute the gradient of the loss with respect to the input
        d_input = np.dot(d_pre_activations, self.weights.T)
        return dW, d_input

class CrossEntropyLoss:
    def __call__(self, y_true, y_pred):
        # Ensure numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred))

    def backward(self, y_true, y_pred):
        # Compute the derivative of the loss with respect to the predicted values
        return -y_true / (y_pred + 1e-15)  # Adding a small value for numerical stability

def forward_pass(inputs, mlp_layers):
    current_input = inputs
    for layer in mlp_layers:
        current_input = layer.forward(current_input)
    return current_input

def MLP_backward(mlp_layers, loss_layer, inputs, targets):
    # Initialize empty dictionaries for storing activations, pre-activations, and weight gradients
    activations = [{} for _ in range(len(mlp_layers))]
    pre_activations = [{} for _ in range(len(mlp_layers))]
    weight_gradients = [{} for _ in range(len(mlp_layers))]

    # Forward pass
    current_input = inputs
    for i, layer in enumerate(mlp_layers):
        pre_activations[i] = layer.forward(current_input)
        activations[i] = layer.activation.call(pre_activations[i])
        current_input = activations[i]

    # Calculate initial error signal
    error_signal = loss_layer.backward(targets, current_input)

    # Backward pass
    for i in reversed(range(len(mlp_layers))):
        d_activation, dW, d_input = mlp_layers[i].backward(error_signal, activations[i])
        error_signal = d_input

        # Store weight gradients in the dictionary
        weight_gradients[i] = dW

    # Update the weights of all the MLP layers
    for i, layer in enumerate(mlp_layers):
        layer.weights -= weight_gradients[i]

    return mlp_layers


# ... (previous code) ...

def train_mlp(mlp_layers, loss_layer, data_generator, epochs, learning_rate, verbose=True):
    loss_values = []
    accuracy_values = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_correct = 0
        num_samples = 0

        for inputs, targets in data_generator:
            # Forward pass
            current_input = inputs
            for layer in mlp_layers:
                current_input = layer.forward(current_input)

            # Calculate loss and accuracy
            epoch_loss += loss_layer(current_input, targets)  # Updated to use the __call__ method
            predictions = current_input
            num_correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))
            num_samples += targets.shape[0]

            # Backward pass
            error_signal = loss_layer.backward(targets, predictions)
            for i in reversed(range(len(mlp_layers))):
                if i == 0:
                    d_activation, dW, d_input = mlp_layers[i].backward_weights(error_signal, inputs)
                else:
                    d_activation, dW, d_input = mlp_layers[i].backward_weights(error_signal, mlp_layers[i - 1].forward(current_input))
                error_signal = d_input

                # Update the weights of the MLP layers
                mlp_layers[i].weights -= learning_rate * dW

        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / num_samples
        accuracy = num_correct / num_samples

        loss_values.append(avg_loss)
        accuracy_values.append(accuracy)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}")

    # Plot the average loss and accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_values, label='Average Loss')
    plt.plot(range(1, epochs + 1), accuracy_values, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Average Loss and Accuracy vs. Epochs')
    plt.legend()
    plt.show()

    return mlp_layers


# ... (remaining code) ...



train_inputs = inputs_train
train_targets = targets_train

# Assuming you have the necessary classes and functions defined

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs, one_hot_targets, test_size=0.2, random_state=42)

# Define the sizes for your MLP layers
input_size = inputs.shape[1]
hidden_size = 32  # Example hidden layer size, replace with your desired size
output_size = one_hot_targets.shape[1]  # Define output size based on the shape of one_hot_targets

# Create instances of the Sigmoid and Softmax classes for activation functions
activation_sigmoid = Sigmoid()
activation_softmax = Softmax()

# Create the MLP layers and the cross-entropy loss function using the defined sizes and activation functions
mlp_layers = [Layer(input_size, hidden_size, activation_sigmoid), Layer(hidden_size, output_size, activation_softmax)]
cross_entropy_loss = CrossEntropyLoss()

# Define the number of epochs and learning rate
epochs = 10
learning_rate = 0.01

# Run the training function with the defined parameters
trained_mlp = train_mlp(
    mlp_layers=mlp_layers,
    loss_layer=cross_entropy_loss,
    data_generator=data_generator(X_train, y_train, minibatch_size=32),
    epochs=epochs,
    learning_rate=learning_rate,
    verbose=True
)

# Test the trained model
test_predictions = []
for i in range(len(X_test)):
    prediction = forward_pass(X_test[i], trained_mlp)
    test_predictions.append(prediction)


# Calculate accuracy
test_predictions = np.array(test_predictions)
predicted_labels = np.argmax(test_predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy: {accuracy}")




