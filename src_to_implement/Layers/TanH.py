import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.X = None
        pass

    def forward(self, input_tensor):
        # Compute the tanh of the input tensor
        self.X = np.tanh(input_tensor)
        return self.X

    def backward(self, error_tensor):
        # Compute the derivative of the tanh and return the product of it with the error tensor
        return error_tensor * (1 - self.X ** 2)
