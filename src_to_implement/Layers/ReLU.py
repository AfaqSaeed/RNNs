import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.X = None
        pass


    def forward(self, input_tensor):
        '''
        apply the ReLU function element-wise on the input tensor
        '''
        # store the inputs
        self.X = input_tensor
        # f(x) = max(0,x)
        # calculate binary mask - element-wise multiplication of input tensor with 0/1
        # 0 for x <= 0
        # x for x > 0
        # elementwise multiplication with binary mask x > 0 -> for x>0 true 1/ for x<= 0 false 0)
        return input_tensor * (input_tensor > 0).astype(float)

    def backward(self, error_tensor):
        # to check, whether error tensor elements are > 0, perform an element-wise multiplication with binary mask
        # bmask = 1 if element of x >= 0, bmask = 0 if element of x < 0 
        # background: derivative of linear activation factor is a constant
        # If x>=0 pass the error_tensor of the previously optimized layer, else block backward pass for specific element
        return error_tensor * (self.X >= 0)