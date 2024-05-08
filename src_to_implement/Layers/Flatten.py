from .Base import BaseLayer
import numpy as np

# flatten layer to reshape a multi-dimensional input to a one dimensional feature vector
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.N = None
        self.image_shape = None
    
    def forward(self, input_tensor):
        # we are using a memory layout, where the rows contain the single samples (images)
        # the subsequent dimensions are the actual shape of the images
        self.input_shape = input_tensor.shape
        # N - Number of samples
        self.N = input_tensor.shape[0]
        # image_shape - Shape of an image passed to the flatten layer
        self.image_shape = input_tensor.shape[1:]
        # input tensor in the shape N x product of all dimensions of the image
        return input_tensor.reshape(self.N, np.prod(self.image_shape))

    def backward(self, error_tensor):
        # put the error tensor in the old shape
        return error_tensor.reshape(self.input_shape)