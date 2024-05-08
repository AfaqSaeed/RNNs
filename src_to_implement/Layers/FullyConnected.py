from .Base import *
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # create a weight tensor with n + 1 rows (1 row for the biases)
        # the weights are part of the adjusted matrix layout -> W' = W.T
        self.weights = np.random.uniform(0, 1, (input_size + 1,output_size))
        self._gradient_weights = np.zeros_like(self.weights)
        self._optimizer = None

    # getter method for optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    # setter method for optimizer property
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    # getter method for gradient weights property
    @property
    def gradient_weights(self):
        # Return the gradients with respect to the weights
        return self._gradient_weights

    def forward(self, input_tensor):
        # Y' = X' * W' = (W*X).T = X.T * W.T-> new memory layout with X', W' being the transposed versions
        # determine the batch size (number of input matrix rows)
        # equals the number of rows of the input tensor (due to other memory layout)
        self.batch_size = input_tensor.shape[0]
        # add a column of ones for the bias for each input row vector (x1,..,xn,1) -> (w1,..,wn,b)
        ones_column = np.ones((self.batch_size, 1))
        input_tensor = np.hstack([input_tensor, ones_column])
        # s tore input tensor for backward pass
        self.X = input_tensor
        # Y' = X' * W' 
        return np.matmul(input_tensor, self.weights)

    def backward(self, error_tensor):
        # calculate the error tensor that is backwards passed to previous layer
        # with adjusted matrix layout: En-1' = En' W'.T
        error_tensor_prev = np.dot(error_tensor, self.weights.T)
        # weight gradient with adjusted matrix layout: W't+1 = W't - learning_rate * X'T*En'
        self._gradient_weights = np.dot(self.X.T, error_tensor)
        # if optimizer is set for this layer, optimize the weights 
        if self._optimizer is not None:
            # weight update: e. g. via Stochastic Gradient Descent
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        # remove bias - last column from error tensor
        # shape returns a tuple with (numRows, numCols) -> tuple[1] = numCols, index of last column = numCols - 1, axis=1 - delte columns
        error_tensor_prev = np.delete(error_tensor_prev,error_tensor_prev.shape[1]-1, axis=1)
        return error_tensor_prev
    
    def initialize(self, weights_initializer, bias_initializer):
        '''
        Method to reinitialize the weights and bias using the provided initializers
        '''
        # determine the shape of the weights for calling the initializer method
        weights_shape = self.weights[:-1, :].shape
        fan_in = weights_shape[0]
        fan_out = weights_shape[1]
        # weight initialization: take all but the the last row of the weights ndarray
        self.weights[:-1, :] = weights_initializer.initialize((fan_in, fan_out), fan_in, fan_out)
        # bias intialization seperately: take the last row of the weights matrix
        bias_shape = self.weights[-1, :].shape
        self.weights[-1, :] = bias_initializer.initialize(bias_shape,1,1)

    def calculate_regularization_loss(self):
        '''
        Method to calculate the regularization loss
        '''
        # calculate the regularization loss
        if self.optimizer is not None:
            return self.optimizer.calculate_regularization_loss(self.weights)
        else:
            return 0
