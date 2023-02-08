import numpy as np


class Dense:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Dense(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )


class Sigmoid:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Sigmoid(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )



class Softmax:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __repr__(self):
        return (
            f"Softmax(input_shape={self.input_shape}, output_shape={self.output_shape})"
        )

