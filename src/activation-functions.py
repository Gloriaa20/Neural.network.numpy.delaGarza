import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(x):
    """
    The Sigmoid function is a sigmoid function that maps any real value
    to a range between 0 and 1. It is commonly used in binary classification problems.
    """
    return 1 / (1 + np.exp(-x))

# Tanh Function
def tanh(x):
    """
    The hyperbolic tangent (Tanh) function maps any real value to a
    range between -1 and 1. It is similar to the Sigmoid function but centered around zero.
    """
    return np.tanh(x)

# ReLU (Rectified Linear Unit) Function
def relu(x):
    """
    The ReLU function sets all negative values to zero, leaving the positive values unchanged.
    It is widely used in deep neural networks.
    """
    return np.maximum(0, x)

# Leaky ReLU Function
def leaky_relu(x, alpha=0.01):
    """
    The Leaky ReLU function is a variation of ReLU that allows a small gradient
    for negative values (controlled by the `alpha` parameter).
    """
    return np.where(x > 0, x, alpha * x)

# Softmax Function
def softmax(x):
    """
    The Softmax function converts a vector of real values into probabilities.
    It is useful in the output layer of a neural network for multi-class classification problems.
    """
    exp_x = np.exp(x - np.max(x))  # Avoid numerical overflow
    return exp_x / np.sum(exp_x, axis=0)

# ELU (Exponential Linear Unit) Function
def elu(x, alpha=1.0):
    """
    The ELU function is similar to ReLU, but for negative values, it returns an
    exponentially decaying function. This helps mitigate the "vanishing gradient" problem.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Swish Function
def swish(x, beta=1):
    """
    The Swish function is a combination of Sigmoid and ReLU. It is a smooth function
    that can overcome some of the limitations of ReLU and Sigmoid.
    """
    return x * sigmoid(beta * x)

# Generate a range of input values for the activation functions
x = np.linspace(-10, 10, 100)

# Calculate the outputs of the different activation functions
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)
y_swish = swish(x)

# Function to plot the activation functions in separate windows
def plot_activation(x, y, title):
    """
    This function takes the input values `x`, the output values `y`, and a title,
    and generates a plot of the activation function in a separate window.
    """
    plt.figure()  # Create a new figure window
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.show()

# Plot each activation function in a separate window
plot_activation(x, y_sigmoid, "Sigmoid")
plot_activation(x, y_tanh, "Tanh")
plot_activation(x, y_relu, "ReLU")
plot_activation(x, y_leaky_relu, "Leaky ReLU")
plot_activation(x, y_elu, "ELU")
plot_activation(x, y_swish, "Swish")