import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.X = None
        pass

    def forward(self, input_tensor):
        # calculate the sigmoid function for each element of the input tensor
        self.X = 1 / (1 + np.exp(-input_tensor))
        return self.X

    def backward(self, error_tensor):
        # calculate the error tensor 
        error_tensor = error_tensor * self.X * (1 - self.X)
        return error_tensor