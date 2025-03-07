# Neural Network Implementation (OR Logic Gate)

This is a simple neural network implementation using Python and NumPy to solve the logical OR problem. The neural network is trained using backpropagation with the sigmoid activation function. The code demonstrates a basic machine learning model for understanding neural networks and their operations.
Table of Contents

    - Project Description
    - Requirements
    - Setup
    - How to Run
    - Code Explanation
    - License


🚀 Project Description

This project implements a feedforward neural network with:

    2 input neurons (representing the two inputs for the OR gate),
    1 hidden layer with 4 neurons,
    1 output neuron that produces the result of the OR gate.

The neural network is trained on the following truth table for the OR operation:
Input 1	Input 2	Output (OR)
0	0	0
0	1	1
1	0	1
1	1	1


🖥️ Requirements

To run this project, you will need:

    Python 3.x
  
    NumPy library

You can install NumPy by running: 

    pip install numpy


📊  Setup

Clone this repository to your local machine:

    git clone <repository_url>
    
    cd <repository_directory>


🛠️ Create a virtual environment (optional but recommended):

    python -m venv venv


Activate the virtual environment:

On Windows:
     
    venv\Scripts\activate

On Mac/Linux:

    source venv/bin/activate


📌  Install the required dependencies:

    pip install -r requirements.txt

(Optional: If you don't have requirements.txt, just install NumPy manually as shown above.)


- How to Run
    Make sure you have the virtual environment activated (if you're using one).
    Run the script: 
    
      python neural_network.py

The network will train for 10,000 epochs and display the error every 1,000 epochs.
After training, the final predictions will be printed, showing the OR gate output for each input.

- Code Explanation
  Neural Network Components:

  Sigmoid Activation Function: The sigmoid function is used as the activation function for both the hidden layer and the output layer. This function maps input values between 0 and 1, making it suitable for binary classification tasks 
  like the OR gate.
  
     def sigmoid(x):
     return 1 / (1 + np.exp(-x))


  Backpropagation: The error from the output is propagated backward to adjust the weights and biases. The derivative of the sigmoid function is used to calculate the gradients for weight updates.

     def sigmoid_derivative(x):
     return x * (1 - x)


Training Loop: The network runs through 10,000 epochs (iterations), performing forward propagation, calculating the error, and updating the weights and biases using the gradient descent algorithm.

  For epoch in range(epochs):
    - Forward propagation and backpropagation steps
    - Update weights and biases


 Output: After training, the model predicts the OR output for each combination of inputs.

    print("\nFinal Predictions:")
    print(predicted_output)

Hyperparameters:
    Learning Rate: 0.1
    Epochs: 10,000
    Number of Hidden Neurons: 4

Weights and Biases:
    Randomly initialized using a uniform distribution to start the training process.
    Adjusted during training via backpropagation.
