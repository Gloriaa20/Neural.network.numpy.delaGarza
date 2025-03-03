# Activation Functions in Neural Networks

This project implements various activation functions used in neural networks and generates graphs to visualize their behavior.


📌 Implemented Functions

Sigmoid: f(x) = 1 / (1 + e^(-x))

Tanh: f(x) = tanh(x)

ReLU (Rectified Linear Unit): f(x) = max(0, x)

Leaky ReLU: f(x) = x if x > 0, alpha * x if x <= 0

Softmax: f(x) = e^x / sum(e^x)

ELU (Exponential Linear Unit): f(x) = x if x > 0, alpha * (e^x - 1) if x <= 0

Swish: f(x) = x * sigmoid(beta * x)


🚀 Requirements

Make sure you have the following dependencies installed before running the code:
pip install numpy matplotlib


🖥️ Usage

Save the code in a file named activations.py and run it with:
python activations.py


📊 Visualization

The code will generate graphs for each activation function, showing their behavior over a range of values.


🛠️ Author and Contribution

Created to demonstrate activation functions in neural networks. Contributions and improvements are welcome! 🚀


# Here’s how you can run the code you provided on your machine:

1. Install Python 3.8.10

- If you don't have Python installed yet, you can download it from the official website.

- Make sure to check the option that says "Add Python to PATH" during installation (this will allow you to run python from the command line).


2. Create a Virtual Environment (Optional but Recommended)

- Creating a virtual environment is a good practice to manage dependencies in your project without affecting other Python installations. To do this:

- Open a terminal or command prompt.

- Navigate to the directory where you want to create your project.

- Run the following command to create a virtual environment:
  python -m venv venv

This will create a folder named venv that contains a separate Python installation.

Activate the virtual environment:
 On Windows: venv\Scripts\activate

 On macOS/Linux: source venv/bin/activate

 Once activated, you’ll see the environment name in your terminal prompt indicating that you're working in the virtual environment.


3. Install the Required Libraries

The code uses numpy and matplotlib. You need to install these libraries if you haven't already. Run the following command to install them (ensure you are inside your virtual environment if you created one):
pip install numpy matplotlib


4. Create the Python File with the Code

Create a new file in your favorite text editor or IDE (e.g., Visual Studio Code, PyCharm, or even a basic text editor). Save the code in a file with the .py extension, for example, activation_functions.py.


5. Run the Code

Once you have the file with the code, you can run it as follows:

- Open a terminal or command prompt.

- Navigate to the directory where you saved the Python file. For example, if you saved it in a folder called activation_functions on your desktop, navigate to that folder:

   On Windows: cd C:\Users\YourUserName\Desktop\activation_functions
   On macOS/Linux: cd ~/Desktop/activation_functions

- Run the Python file with: python activation_functions.py


6. View the Results

Once you run the code, three plots will be generated showing the activation functions Sigmoid, Tanh, and ReLU. The plots will appear one by one in a Matplotlib window.

- Summary of Steps:
   Install Python.
   Create a virtual environment (optional).
   Install the necessary libraries (numpy and matplotlib).
   Create a .py file with the code.
   Run the Python file using python file.py.

After following these steps, you should be able to see the plots generated for the activation functions. 



# if you have errors, follow the commands.

pip install numpy
pip3 install numpy

source env/bin/activate
pip install matplotlib
python -c "import matplotlib; print(matplotlib.__version__)"
python functions.py

which pip  
pip install --upgrade pip

