import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_prime(Z):
    return np.greater_equal(Z, 0).astype(int)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


class Dense:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Dense(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

    def initialize_parameters(self):
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)

    def forward(self, input):
        self.input = input
        self.linear_output = np.dot(self.weights, self.input) + self.bias
        self.output = relu(self.linear_output)

        return self.output

    def backward(self, dpartial):
        m = dpartial.shape[1]

        dZ = dpartial * relu_prime(self.linear_output)
        dW = np.dot(dZ, self.input.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(self.weights.T, dZ)

        return dW, db, dA

    def update_parameters(self, learning_rate, dW, db):
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * db


class Sigmoid:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Sigmoid(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

    def initialize_parameters(self):
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)

    def forward(self, input):
        self.input = input
        Z = np.dot(self.weights, self.input) + self.bias
        self.output = sigmoid(Z)

        return self.output

    def backward(self, Y):
        m = Y.shape[1]

        dZ = self.output - Y
        dW = np.dot(dZ, self.input.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(self.weights.T, dZ)

        return dW, db, dA

    def update_parameters(self, learning_rate, dW, db):
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * db


class Softmax:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Softmax(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

    def initialize_parameters(self):
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)

