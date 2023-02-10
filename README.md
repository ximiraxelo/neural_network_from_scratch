# Neural Network from scratch 🧠

This project is inspired by the [Andrew Ng's Deep Learning course on Coursera](https://www.coursera.org/learn/neural-networks-deep-learning).

## Objective 🎯

The objective of this project is to implement a Neural Network for binary classification from scratch. NumPy was used for vectorization and matrix manipulations.

## Project 🐍

The project is built in Python 3.10.2 with NumPy.

## Getting started 💻

### Setup

```python
from src.layers import Dense, Sigmoid
from src.main import NeuralNetwork
```

### Layers

Let's look at the Dense layer:

```python
dense_layer = Dense(input_shape=4, output_shape=16)
```

The input shape represents the number of features ($n_x$) in the dataset and the output shape the number of hidden units. This layer uses the ReLU (Rectified Linear Unit) activation function.

In the Sigmoid layer we pass only the `input_shape`, because the sigmoid output shape is always 1.

```python
sigmoid_layer = Sigmoid(input_shape=16)
```

The sigmoid layer is always the last layer in the model.

### Model

Now we can create a model with sequential layers

```python
model = NeuralNetwork([
    Dense(4, 16),
    Dense(16, 16),
    Sigmoid(16)
])
```

Look that the output shape of any layer is the same as the input shape of the next layer.

### Training the model

The model can be trained with the `NeuralNetwork.fit()` method

```python
model.fit(x_train, y_train, epochs=100, learning_rate=0.01, print_step=10)
```

### Predicting

We can predict after the training using the `NeuralNetwork.predict()` method

```python
prediction = model.predict(x_test)
```

### Evaluating the model

The model can be evaluated with the `NeuralNetwork.evaluate()` method

```python
accuracy = model.evaluate(x_test, y_test)
```

This method returns the accuracy (in the range [0, 1]) on the test set.

## Tests 🧪

Two simple models were created for binary classification on the Bank Note Authentication dataset and the Iris dataset. Check the Notebooks on the root of this project.