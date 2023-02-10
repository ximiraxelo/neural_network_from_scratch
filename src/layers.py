import numpy as np
import numpy.typing as npt


def relu(Z: npt.NDArray) -> npt.NDArray:
    """Computes the ReLU (Rectified Linear Unit) activation function of a
    given input.

    Args:
        Z (npt.NDArray): Linear input.

    Returns:
        npt.NDArray: Output of the ReLU activation function.
    """
    return np.maximum(0, Z)


def relu_prime(Z: npt.NDArray) -> npt.NDArray:
    """Computes the firs derivate of the ReLU (Rectified Linear Unit).

    Args:
        Z (npt.NDArray): Linear input.

    Returns:
        npt.NDArray: Output of the first derivative of ReLU.
    """
    return np.greater_equal(Z, 0).astype(int)


def sigmoid(Z: npt.NDArray) -> npt.NDArray:
    """Computes the sigmoid activation function of a given input.

    Args:
        Z (npt.NDArray): Linear input.

    Returns:
        npt.NDArray: Output of the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-Z))


class Dense:
    def __init__(self, input_shape: int, output_shape: int) -> None:
        """Dense layer with ReLU activation function.

        Args:
            input_shape (int): Shape of the input, number of the units
            of the previous layer.
            output_shape (int): Number of units
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self) -> str:
        return (
            f"Dense(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

    def initialize_parameters(self) -> None:
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)

    def forward(self, input: npt.NDArray) -> npt.NDArray:
        """Computes the forward propagation of the layer.

        Args:
            input (npt.NDArray): Input of the layer, output of the previous layer.

        Returns:
            npt.NDArray: Output of the layer.
        """
        self.input = input
        self.linear_output = np.dot(self.weights, self.input) + self.bias
        self.output = relu(self.linear_output)

        return self.output

    def backward(
        self, dpartial: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Computes the backward propagation of the layer.

        Args:
            dpartial (npt.NDArray): dA of the next layer.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: dW: derivative of
            loss function in respect to weights, db: derivative of the
            loss function in respect to bias, dA: derivative of the
            loss function in respect of A.
        """
        m = dpartial.shape[1]

        dZ = dpartial * relu_prime(self.linear_output)
        dW = np.dot(dZ, self.input.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(self.weights.T, dZ)

        return dW, db, dA

    def update_parameters(
        self, learning_rate: float, dW: npt.NDArray, db: npt.NDArray
    ) -> None:
        """Update the layer parameters using Gradient Descent.

        Args:
            learning_rate (float): Influence how much the
            neural network parameters will be updated.
            dW (npt.NDArray): Derivative of the loss function in respect of weights.
            db (npt.NDArray): Derivative of the loss function in respect of bias.
        """
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * db


class Sigmoid:
    def __init__(self, input_shape: int) -> None:
        """Dense layer with Sigmoid activation function.

        Sigmoid will always be the last layer.

        Args:
            input_shape (int): Shape of the input, number of the units
            of the previous layer.
        """
        self.input_shape = input_shape
        self.output_shape = 1

    def __repr__(self) -> str:
        return (
            f"Sigmoid(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

    def initialize_parameters(self) -> None:
        self.weights = np.random.randn(self.output_shape, self.input_shape)
        self.bias = np.random.randn(self.output_shape, 1)

    def forward(self, input: npt.NDArray) -> npt.NDArray:
        """Computes the prediction.

        Args:
            input (npt.NDArray): Input of the layer, output of the previous layer.

        Returns:
            npt.NDArray: Prediction of the neural network.
        """
        self.input = input
        Z = np.dot(self.weights, self.input) + self.bias
        self.output = sigmoid(Z)

        return self.output

    def backward(self, Y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Computes the backward propagation of the layer.

        Args:
            Y (npt.NDArray): Target data.

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: dW: derivative of
            loss function in respect to weights, db: derivative of the
            loss function in respect to bias, dA: derivative of the
            loss function in respect of A.
        """
        m = Y.shape[1]

        dZ = self.output - Y
        dW = np.dot(dZ, self.input.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(self.weights.T, dZ)

        return dW, db, dA

    def update_parameters(
        self, learning_rate: float, dW: npt.NDArray, db: npt.NDArray
    ) -> None:
        """Update the layer parameters using Gradient Descent.

        Args:
            learning_rate (float): Influence how much the
            neural network parameters will be updated.
            dW (npt.NDArray): Derivative of the loss function in respect of weights.
            db (npt.NDArray): Derivative of the loss function in respect of bias.
        """
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * db
