import numpy as np
class BaseOptimizer:
    def __init__(self) -> None:
        self.regularizer = None

    def calculate_regularization_loss(self, weights):
        '''
        Method to calculate the regularization loss
        '''
        regularization_loss = 0
        if self.regularizer is not None:
            regularization_loss = self.regularizer.norm(weights)
        return regularization_loss

    def add_regularizer(self,regularizer):
        self.regularizer  = regularizer
# stochastic gradient descent updates the weigths after each iteration
class Sgd(BaseOptimizer):
    def __init__(self, learning_rate : float):
        super().__init__()
        '''
        define a stochastic gradient descent instance with a given learning rate (step size)
        '''
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        return the updated tensor: current_weight_tensor - gradient_tensor * learning_rate
        '''
        if self.regularizer is not None:
            # add regularization term as a constraint
            return weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)) - self.learning_rate * gradient_tensor
        else:
            return weight_tensor - self.learning_rate * gradient_tensor
    
class SgdWithMomentum(BaseOptimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__()
        '''
        Define a stochastic gradient descent with momentum instance.
        '''
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0  # Initialize velocity to zero

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
        Return the updated tensor using SGD with momentum update rule.
        '''
        self.velocity = self.momentum_rate * self.velocity + self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            # add regularization term as a constraint
            return weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)) - self.velocity
        else:
            return weight_tensor - self.velocity


class Adam(BaseOptimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__()
        '''
        Define an Adam optimizer instance.
        '''
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0  # Initialize first moment to zero
        self.r = 0  # Initialize second moment to zero
        self.k = 0  # Initialize time step to zero
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        '''
            Return the updated tensor using the Adam optimizer update rule.
        '''
        # Increment the iteration count
        self.k += 1

        # Calculate the exponentially decaying average of past gradients
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor

        # Calculate the exponentially decaying average of past squared gradients
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor**2

        # Correct bias in the first moment estimate
        v_hat = self.v / (1 - self.mu**self.k)

        # Correct bias in the second raw moment estimate
        r_hat = self.r / (1 - self.rho**self.k)

        # Calculate the update to the weights
        update = self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)

        # Return the updated weights
        if self.regularizer is not None:
            # add regularization term as a constraint
            return weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)) - update
        else:  
            return weight_tensor - update