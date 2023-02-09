import numpy as np

from layers import Dense, Sigmoid, Softmax


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.n_layers = len(layers)

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers})"

    def initialize_parameters(self):
        for layer in self.layers:
            layer.initialize_parameters()

