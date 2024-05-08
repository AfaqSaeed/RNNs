

import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride, pooling_shape):
        super().__init__()
        self.stride = stride
        self.pooling_shape = pooling_shape

        self.max_indeces = None
        self.input_shape = None

    def forward(self, input_tensor):
        # Store the input shape for later use in the backward pass
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        self.max_indeces = {}  # Dictionary to store the indices of max values for each pooled region
        self.out_shape = []

        # Calculate the output shape after pooling
        for input_dim, pooling_dim, stride in zip(input_tensor.shape[2:], self.pooling_shape, self.stride):
            rounded = np.int64(np.floor(np.float64(input_dim - pooling_dim) / np.float64(stride)) + 1)
            self.out_shape.append(rounded)

        self.out_shape = (self.input_shape[1], *self.out_shape)

        # Initialize the result tensor with zeros
        result = np.zeros((batch_size, *self.out_shape))

        # Iterate through each element in the input tensor
        for batch_index in range(batch_size):
            for channel_index in range(self.out_shape[0]):
                for output_first_dimension in range(self.out_shape[1]):
                    input_first_dimension = self.stride[0] * output_first_dimension
                    for output_second_dimension in range(self.out_shape[2]):
                        input_second_dimension = self.stride[1] * output_second_dimension

                        # Extract the current pooling region
                        input_part = input_tensor[
                                     batch_index,
                                     channel_index,
                                     input_first_dimension:input_first_dimension + self.pooling_shape[0],
                                     input_second_dimension:input_second_dimension + self.pooling_shape[1]]

                        # Find the indices of the maximum value in the pooling region
                        max_location = np.unravel_index(indices=input_part.argmax(), shape=input_part.shape)

                        # Save the indices in the max_indeces dictionary
                        output_location = (batch_index, channel_index, output_first_dimension, output_second_dimension)
                        self.max_indeces[output_location] = (max_location[0] + input_first_dimension, max_location[1] + input_second_dimension)

                        # Assign the maximum value to the result tensor
                        result[output_location] = input_part[max_location]

        # Additional comment: Return the result tensor after max pooling
        return result

    def backward(self, error_tensor):
        # Initialize a tensor to accumulate the errors below
        error_below = np.zeros(self.input_shape)

        # Iterate through the stored max indices and accumulate errors in the corresponding positions
        for complete_index, max_location in self.max_indeces.items():
            shape = (complete_index[0], complete_index[1], *max_location)
            error_below[shape] += error_tensor[complete_index]

        # Return the accumulated errors for the backward pass
        return error_below
