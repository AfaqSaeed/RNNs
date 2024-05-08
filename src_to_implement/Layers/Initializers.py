import numpy as np

class Constant:
    def __init__(self, const_value=0.1):
        self.const_value = const_value

    def initialize(self, shape, fan_in, fan_out):
        # Return an initialized tensor with a constant value
        return np.full(shape, self.const_value)

class UniformRandom:
    def initialize(self, shape, fan_in, fan_out):
        '''      
        Return an initialized tensor with values between 0 and 1 from a uniform distribution
        '''        
        return np.random.uniform(0, 1, size=shape)

class Xavier:
    def initialize(self, shape, fan_in, fan_out):
        '''
        Return an initialized tensor with values from a normal distribution (Xavier/Glorot initialization)
        '''        
        # normalization with number of input elements and output elements
        # the more inputs there are, the higher the possible output of the distribution and thus the variance -> normalize variance with fan_in and fan_out
        sigma = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, sigma, size=shape)

class He:
    def initialize(self, shape, fan_in, fan_out):
        '''
        Return an initialized tensor with values from a normal distribution (He initialization)
        '''
        # normalization with number of input elements
        # the more inputs there are, the higher the variance of the distribution -> normalize with fan_in
        sigma = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, sigma, size=shape)
