import numpy as np

# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input data (X) and expected labels (y)
# In this case, we are working with a simple logical OR problem
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [1]])

# Define neural network parameters
input_layer_neurons = X.shape[1]  # Number of neurons in the input layer (2 features)
hidden_layer_neurons = 4         # Number of neurons in the hidden layer
output_layer_neurons = 1         # Number of neurons in the output layer (1 output)

# Random initialization of weights and biases for each layer
np.random.seed(42)  # For reproducibility

# Weights from input layer to hidden layer
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))

# Weights from hidden layer to output layer
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bias_output = np.random.uniform(size=(1, output_layer_neurons))

# Learning rate
learning_rate = 0.1

# Number of iterations (epochs) for training the neural network
epochs = 10000

# Neural network training
for epoch in range(epochs):
    # Forward propagation
    # Hidden layer
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Output layer
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = y - predicted_output

    # Backpropagation
    # Calculate the derivative of the error with respect to the output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Calculate the error in the hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update the weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Show the error every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

# Final predictions after training
print("\nFinal Predictions:")
print(predicted_output)
