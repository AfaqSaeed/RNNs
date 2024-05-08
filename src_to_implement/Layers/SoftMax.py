import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.Y = None
        pass

    def forward(self, input_tensor):
        '''
        apply the softmax function
        '''
        # stability criterion from description
        # x_stable = input elements of x - maximum of input tensor
        X_stable = input_tensor - np.max(input_tensor)
        # softmax nominator
        nominator = np.exp(X_stable)
        # softmax denominator -> sum of all elements of the row (one sample from batch)
        denominator = np.sum(nominator, axis=1)
        denominator = np.reshape(denominator, (denominator.shape[0],1))
        # softmax 
        Y_column = nominator / denominator
        # store output for backward pass
        self.Y = Y_column
        return Y_column

    def backward(self, error_tensor):
        '''
        backward pass for softmax function
        '''
        # sum of error tensor multiplied with with softmax prediction
        error_sum = np.sum(error_tensor*self.Y, axis=1)
        # reshape row vector to column vector
        error_sum_row = error_sum.reshape(error_sum.shape[0],1)
        # element wise multiplication of predicted value and error term
        error_tensor_prev = self.Y * (error_tensor - error_sum_row)
        return error_tensor_prev
