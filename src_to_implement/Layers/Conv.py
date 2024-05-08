import numpy as np
from .Base import BaseLayer
from copy import deepcopy
from .Initializers import *
from scipy.signal import convolve, correlate

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape , num_kernels : int) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        
        if len(convolution_shape) == 3:
            self.convolution_shape = convolution_shape
        else:
            # if only 1D kernel is provided, store it as the row dimension
            self.convolution_shape = (convolution_shape[0], 1, convolution_shape[1])

        if len(stride_shape) == 2:
            # if both dimensions of the stride are provided, story them
            self.stride = stride_shape
        else: 
            # if only one stride dimension is provided, select 1 as the row stride
            self.stride = (1,stride_shape)

        # number of kernels determines how many different kernels are applied in the conv layer
        self.num_kernels = num_kernels
        # the weigths are the parameters of the different kernels
        # each kernel has the same size 
        self.weights_shape = (num_kernels, *self.convolution_shape)

        # initialize weights randomly 
        self.weights = np.random.random(self.weights_shape)
        # create bias for each filter kernel
        self.bias = np.random.random(num_kernels)
           
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._optimizer_bias = None
    
    def initialize(self, weights_initializer, bias_initializer): 
        # calculate number of inputs & outputs
        # number of inputs = number of kernel components    
        fan_in = np.prod(self.convolution_shape)
        # number of outputs = number of kernels * number of output components except channels (channels are flattened by convolution)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])     
        # initialize the weights and biases with given initializers
        self.weights = weights_initializer.initialize(self.weights_shape, fan_in, fan_out)
        # for bias fan_in / fan_out not considered (e. g. constant initialization)
        self.bias = bias_initializer.initialize(self.num_kernels, 1, 1)

    # getter method for optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    # setter method for optimizer property
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_bias = deepcopy(optimizer)
    # getter method for gradient weights property
    @property
    def gradient_weights(self):
        # Return the gradients with respect to the weights
        return self._gradient_weights
    # getter method for gradient weights property
    @property
    def gradient_bias(self):
        # Return the gradients with respect to the weights
        return self._gradient_bias
    
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.single_input_shape = input_tensor.shape[1:]
        self.single_input_shape = self.single_input_shape if len(self.single_input_shape) == 3 else (self.single_input_shape[0], 1, self.single_input_shape[1])
        
        stride_y, stride_x = self.stride
        self.input_tensor = input_tensor.reshape(-1,*(self.single_input_shape))

        output_list = []
        # iterate over each kernel of the layer
        for k in range(self.num_kernels):
            # add an extra dimension at the beginning of the weights array (for the batch dimension)
            kernel = np.expand_dims(self.weights[k], 0)
            # correlate whole batch with the kernel
            result = correlate(input_tensor, kernel, mode='same')
            # After the correlation, the result tensor is subsampled:
            # The second dimension (result.shape[1]) corresponds to the height.
            # int(np.floor(result.shape[1] / 2)) selects the central row of 
            # ::stride_y and ::stride_x perform subsampling along the y and x axes. -> Hint 3 (subsampling from full convolution)
            result = result[:, int(np.floor(result.shape[1] / 2)), ::stride_y, ::stride_x]
            # add a bias for each kernel
            result += self.bias[k]
            # append the output tensor to the list 
            output_list.append(result)
        # create a ndarray from the output list -> add the correlated 
        output_tensor = np.stack(output_list, 1)
        self.output_shape = output_tensor.shape

        return output_tensor
    
    def backward(self, error_matrix):

        # Reshape the error matrix to match the output shape of the convolutional layer
        error_matrix = error_matrix.reshape(self.output_shape)
        
        # Pad the error matrix to account for the convolutional stride
        error_matrix_dim_changed = np.zeros((error_matrix.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]), dtype=error_matrix.dtype)
        error_matrix_dim_changed[..., ::self.stride[0], ::self.stride[1]] = error_matrix
        error_matrix = error_matrix_dim_changed

        # Calculate the gradient for the bias by summing values along specified axes
        self._gradient_bias = np.sum(error_matrix, axis=(0, 2, 3))

        # Calculate the derivative of the input tensor with respect to the error
        derivativeofx = []
        for input_channels in range(self.input_tensor.shape[1]):
            kernel = []
            for output_channels in range(0, self.num_kernels):
                # Extract the weights for a specific input channel and output channel
                kernel.append(self.weights[output_channels][input_channels])
            # Stack the kernel, reverse it, and add dimensions to match convolution operation
            kernel = np.stack(np.expand_dims(kernel[::-1], 1), 1)
            # Perform convolution operation to compute the derivative of the input
            derivativeofx.append(convolve(error_matrix, kernel, 'same')[:, int(kernel.shape[1] / 2)])
        derivativeofx = np.stack(derivativeofx, 1)

        # Calculate the left padding for the x-axis based on half of the convolution filter width
        left_pad_of_x = int(self.convolution_shape[2] / 2)

        # Check if the convolution filter width is an odd number
        if self.convolution_shape[2] % 2 != 0:
            # If odd, set right padding to the same value as left padding
            right_pad_of_x = int(self.convolution_shape[2] / 2)
        else:
            # If even, set right padding to one less than the left padding
            right_pad_of_x = int(self.convolution_shape[2] / 2) - 1

        # Calculate the left padding for the y-axis based on half of the convolution filter height
        left_pad_of_y = int(self.convolution_shape[1] / 2)

        # Check if the convolution filter height is an odd number
        if self.convolution_shape[1] % 2 != 0:
            # If odd, set right padding to the same value as left padding
            right_pad_of_y = int(self.convolution_shape[1] / 2)
        else:
            # If even, set right padding to one less than the left padding
            right_pad_of_y = int(self.convolution_shape[1] / 2) - 1

        # Calculate the gradient for the weights
        # Initialize an empty list to store gradient weights for each kernel
        gradient_weights = []

        # Pad the input tensor based on the calculated left and right padding values
        input_padded = np.pad(self.input_tensor, ((0, 0), (0, 0), (left_pad_of_y, right_pad_of_y), (left_pad_of_x, right_pad_of_x)), 'constant')

        # Loop over each output channel (kernel)
        for h in range(0, self.num_kernels):
            # Initialize an empty list to store derivatives for each input channel
            derivativeofwidthheight = []
            
            # Loop over each input channel
            for second_dim_no in range(0, self.input_tensor.shape[1]):
                # Perform convolution operation on the padded input tensor
                # This is done by correlating the input channel with the error matrix for a specific kernel
                derivativeofwidthheight.append(correlate(input_padded[:, second_dim_no], error_matrix[:, h], 'valid'))
            
            # Stack the derivatives for each input channel to form a 2D array
            gradient_weights.append(np.stack(derivativeofwidthheight, 1))

        # Stack the gradient weights for each kernel to form a 3D array
        gradient_weights = np.stack(gradient_weights, 1)

        # Remove singleton dimensions from the weight gradients
        self._gradient_weights = np.squeeze(gradient_weights, axis=(0,))

        # Update weights and biases using the optimizer if provided
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        # Adjust the shape of the output derivative if needed
        derivativeofx = derivativeofx if len(self.single_input_shape) == 3 else derivativeofx[:, :, 0, :]

        # Return the derivative of the input tensor with respect to the error
        return derivativeofx

       
    def calculate_regularization_loss(self):
        '''
        Method to calculate the regularization loss
        '''
        # calculate the regularization loss
        if self.optimizer is not None:
            return self.optimizer.calculate_regularization_loss(self.weights)+ self.optimizer.calculate_regularization_loss(self.bias)
        else:
            return 0
