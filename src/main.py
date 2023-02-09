import numpy as np
import numpy.typing as npt
from layers import Dense, Sigmoid


class NeuralNetwork:
    def __init__(self, layers: list[Dense | Sigmoid]) -> None:
        self.layers = layers
        self.n_layers = len(layers)

    def __repr__(self) -> str:
        return f"NeuralNetwork(layers={self.layers})"

    def initialize_parameters(self) -> None:
        for layer in self.layers:
            layer.initialize_parameters()

    def forward_propagation(self, x_train: npt.NDArray) -> npt.NDArray:
        A = x_train

        for layer in self.layers:
            A = layer.forward(A)

        return A

    def backward_propagation(self, y_train: npt.NDArray, learning_rate: float) -> None:
        dA = y_train

        for layer in reversed(self.layers):
            dW, db, dA = layer.backward(dA)
            layer.update_parameters(learning_rate, dW, db)

    def cost(self, predicted: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        m = y.shape[1]
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1-epsilon)
        cost = (
            -(np.dot(y, np.log(predicted).T) + np.dot(1 - y, np.log(1 - predicted).T))
            / m
        )

        return np.squeeze(cost)

    def fit(
        self,
        x_train: npt.NDArray,
        y_train: npt.NDArray,
        epochs: int,
        learning_rate: float,
        print_step: int = 100,
    ) -> None:
        self.initialize_parameters()

        for epoch in range(epochs):
            predicted = self.forward_propagation(x_train)
            self.backward_propagation(y_train, learning_rate)

            if ((epoch % print_step) == 0) or (epoch == (epochs - 1)):
                cost = self.cost(predicted, y_train)
                print(f"Epoch: {epoch}, Cost: {cost}\n")

    def predict(self, x_test: npt.NDArray) -> npt.NDArray | np.float64:
        predicted = self.forward_propagation(x_test)
        predicted = np.round(predicted)

        return predicted

    def evaluate(self, x_test: npt.NDArray, y_test: npt.NDArray) -> np.float64:
        predictions = self.predict(x_test)
        correct_predictions = np.sum(predictions == y_test)
        accuracy = correct_predictions / y_test.shape[1]

        return accuracy
