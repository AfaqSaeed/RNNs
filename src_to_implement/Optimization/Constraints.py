import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # Calculate L1 regularization gradient
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # Calculate L1 regularization norm
        return self.alpha * np.sum(np.abs(weights))

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # Calculate L2 regularization gradient
        return self.alpha * weights

    def norm(self, weights):
        # Calculate L2 regularization norm
        # calculate the frobesius norm of the weights
        return self.alpha * np.sum(np.square(weights))


