import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers})"

    def initialize_parameters(self):
        for layer in self.layers:
            layer.initialize_parameters()

    def forward_propagation(self, x_train):
        A = x_train

        for layer in self.layers:
            A = layer.forward(A)

        return A

    def backward_propagation(self, y_train, learning_rate):
        dA = y_train

        for layer in reversed(self.layers):
            dW, db, dA = layer.backward(dA)
            layer.update_parameters(learning_rate, dW, db)

