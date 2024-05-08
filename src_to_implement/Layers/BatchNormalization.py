import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        """
        Constructor for the Batch Normalization layer.

        Args:
            channels (int): Number of channels of the input tensor.
        """
        self.channels = channels
        # beta = bias, gamma = weights
        self.bias = None
        self.weights = None
        self.initialize()
        self.trainable = True
        self.input_convolutional = False
        self.alpha = 0.8
        self.moving_mean = None
        self.moving_var = None
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    def initialize(self,weights=None,bias=None):
        """
        Method to initialize the bias beta and the weights gamma.
        """
        # Initialize beta and gamma using the channels size
        self.bias = np.zeros(self.channels) # Initialize bias beta with 0 
        self.weights = np.ones(self.channels) # Initialize gamma weights
    
    def forward(self, input_tensor):
        """
        Forward pass of the BatchNormalization layer.

        Args:
            input_tensor (ndarray): Input tensor.

        Returns:
            ndarray: Output tensor after applying BatchNormalization.
        """
        if len(input_tensor.shape) == 4:
            self.input_convolutional = True
        else:
            self.input_convolutional = False
        if self.input_convolutional:
            input_tensor=self.reformat(input_tensor)
        self.X = input_tensor
        batch_mean = np.mean(input_tensor, keepdims=True,axis=0)
        batch_var = np.var(input_tensor, keepdims=True, axis=0)
        if self.moving_mean is None and self.moving_var is None:
            self.moving_mean = batch_mean.copy()
            self.moving_var = batch_var.copy()
        if self.testing_phase:
            batch_mean = self.moving_mean
            batch_var = self.moving_var
        else:
            self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * batch_mean
            self.moving_var = self.alpha * self.moving_var + (1 - self.alpha) * batch_var
        denominator = np.sqrt(batch_var + 1e-10)
        numerator = input_tensor - batch_mean
        self.transformed_input = numerator / denominator
        output = self.weights * (numerator / denominator) + self.bias
        if self.input_convolutional:
            output = self.reformat(output)
        return output
            
             
    def backward(self, error_tensor):
        """
        Backward pass of the BatchNormalization layer.

        Args:
            error_tensor (ndarray): Error tensor.

        Returns:
            ndarray: Error tensor after backpropagation.
        """
        if self.input_convolutional:
            error_tensor=self.reformat(error_tensor)
        self._gradient_bias = np.sum(error_tensor, axis=0)
        self._gradient_weights = np.sum(error_tensor * self.transformed_input, axis=0)
        if self._optimizer is not None:
            self.bias = self._optimizer.calculate_update(self.bias, self._gradient_bias)
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        error_tensor = compute_bn_gradients(error_tensor, self.X, self.weights, self.moving_mean, self.moving_var)
        if self.input_convolutional:
            error_tensor = self.reformat(error_tensor)
        return error_tensor
    
    def reformat(self, input_tensor):   
        if len(input_tensor.shape) == 4:
            self.b, self.h, self.m, self.n = input_tensor.shape
            # Flatten the input tensor, keep the batch and channel dimension
            input_tensor = input_tensor.reshape((self.b,self.h, -1))
            input_tensor = input_tensor.transpose(0,2,1)
            # flatten the input tensor, keep the channel dimension
            output_tensor = input_tensor.reshape(input_tensor.shape[0]*input_tensor.shape[1],input_tensor.shape[2])
        if len(input_tensor.shape) == 2:
            input_tensor= input_tensor.reshape(self.b,  self.m* self.n, self.h)    
            input_tensor = input_tensor.transpose(0,2,1)
            output_tensor = input_tensor.reshape(self.b, self.h, self.m, self.n)
        return output_tensor

# getter method for optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    # setter method for optimizer property
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        # Return the gradients with respect to the weights
        return self._gradient_weights
    
    @property
    def gradient_bias(self):
        # Return the gradients with respect to the bias
        return self._gradient_bias