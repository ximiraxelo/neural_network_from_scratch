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

