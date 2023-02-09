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

    def cost(self, predicted, y):
        m = y.shape[1]
        cost = (
            -(np.dot(y, np.log(predicted).T) + np.dot(1 - y, np.log(1 - predicted).T))
            / m
        )

        return np.squeeze(cost)

    def fit(self, x_train, y_train, epochs, learning_rate, print_step=100):
        self.initialize_parameters()

        for epoch in range(epochs):
            predicted = self.forward_propagation(x_train)
            self.backward_propagation(y_train, learning_rate)

            if ((epoch % print_step) == 0) or (epoch == (epochs - 1)):
                cost = self.cost(y_train)
                print(f"Epoch: {epoch}, Cost: {cost}\n")

    def predict(self, x_test):
        predicted = self.forward_propagation(x_test)
        predicted = np.round(predicted)

        return predicted

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct_predictions = np.sum(predictions == y_test)
        accuracy = correct_predictions / y_test.shape[1]

        return accuracy
