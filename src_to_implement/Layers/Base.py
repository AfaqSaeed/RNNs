class BaseLayer:
    def __init__(self):
        '''
        This class will be inherited by every layer in our framework.
        '''
        # flag to tell, whether layer can be trained (e. g. fully connected layer) or not (e. g. ReLU activation layer)
        self.trainable = False
        self.testing_phase = False

    def calculate_regularization_loss(self):
        '''
        Return 0 as regularization loss for base layer
        '''
        return 0