import numpy as np 
from .Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, dropout_probability):
        super().__init__()
        self.trainable = False
        self.dropout_probability = dropout_probability

    def forward(self,input_tensor):
        if not self.testing_phase :
            # Generate a binary mask with values less than dropout_probability
            self.set_to_zero_array = np.random.rand(*input_tensor.shape) < self.dropout_probability
            # Apply the mask to the input tensor
            output_tensor = input_tensor * self.set_to_zero_array
            # Scale the output tensor to maintain the expected value
            output_tensor = output_tensor / self.dropout_probability 
        
            return output_tensor
        else: 
            return input_tensor

    def backward(self,error_tensor):
        if not self.testing_phase:    
            # Apply the mask to the error tensor
            output_tensor = error_tensor*self.set_to_zero_array
            # Scale the output tensor to maintain the expected value
            output_tensor = output_tensor/self.dropout_probability 
            return output_tensor
