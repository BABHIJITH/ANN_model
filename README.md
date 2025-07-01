🧠 Artificial Neural Network (ANN) Project
This repository presents a hands-on demonstration of Artificial Neural Networks (ANNs) using the MNIST dataset. It covers:

A basic ANN implementation

A deeper Feedforward Neural Network (FNN)

A breakdown of the Backpropagation algorithm

📚 Table of Contents
Introduction to ANNs

Project Structure

Getting Started

3.1 Prerequisites

3.2 Installation

3.3 Usage

Code Overview

4.1 Basic ANN Implementation

4.2 Feedforward Neural Network (FNN)

4.3 Backpropagation

Results

Conclusion

1. Introduction to ANNs
Artificial Neural Networks mimic the human brain's neural structure. They're composed of interconnected layers of nodes that learn to recognize complex data patterns.

🔍 Key Components:
Input Layer: Receives raw data (e.g., image pixels)

Hidden Layers: Apply transformations and nonlinear functions

Output Layer: Produces final predictions or classifications

🧠 Core Mechanics:
Weighted neuron connections

Activation functions like ReLU, Sigmoid, or Tanh

Training via Backpropagation using Gradient Descent

🔧 Common Applications:
Image and speech recognition

Financial forecasting

Medical diagnostics

Natural language processing

2. Project Structure
ANN_DEEPL_assignment.ipynb — A Jupyter Notebook with full implementations and explanations

3. Getting Started
3.1 Prerequisites
Install the following Python libraries:

tensorflow

keras (included with TensorFlow)

numpy

3.2 Installation

pip install tensorflow numpy

3.3 Usage

git clone https://github.com/BABHIJITH/ANN-Project.git

cd ANN-Project

jupyter notebook ANN_DEEPL_assignment.ipynb

Run the notebook cells step-by-step to explore ANN, FNN, and backpropagation examples.

4. Code Overview

4.1 Basic ANN Implementation
Dataset: Loaded via mnist.load_data()

Model:

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
Training: 5 epochs using Adam optimizer and sparse categorical crossentropy

Evaluation: Accuracy measured on test dataset

4.2 Feedforward Neural Network (FNN)
Demonstrates a deeper neural network with one-directional data flow.

Model Structure:

Flatten(input_shape=(28, 28))
Dense(128, activation='relu')
Dense(64, activation='relu')
Dense(64, activation='relu')
Dense(10, activation='softmax')
Preprocessing:

Normalize input images

Convert labels using to_categorical

Compilation: Adam optimizer with categorical crossentropy

4.3 Backpropagation
Backpropagation is essential for training neural networks effectively.

🔄 Training Steps:
Forward Pass: Input flows through the network

Loss Calculation: Compare output vs true label

Backward Pass: Compute gradients via chain rule

Weight Update: Adjust weights using optimizer (e.g., Adam)

A model identical to the FNN is trained using model.fit() to apply backpropagation.

5. Results
Model	Test Accuracy
Basic ANN	~0.9396
Feedforward Neural Network	~0.0783
Backpropagation-trained FNN	~0.9754

⚠️ The FNN model shows low initial accuracy as it was evaluated before training.

6. Conclusion
This project offers a foundational understanding of how ANNs work, with practical insights into Feedforward networks and the critical role of Backpropagation. The final model demonstrates high accuracy in digit recognition, showcasing the power of deep learning.
