import numpy as np
import copy

class NeuralNetwork:
    
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        # Initialize NN with selected optimizer
        self.phase = "Train"  # Adding String so that we can change it from train to test
        self.optimizer = optimizer
        self.loss = []  # List to store loss during training
        self.layers = []  # List to store layers of the neural network
        self.data_layer = None  # Placeholder for a data layer
        self.loss_layer = None  # Placeholder for a loss layer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.label_tensor = None  # Placeholder for the label tensor
        self.optimizer = optimizer  # New optimizer label variable

    def append_layer(self, layer):
        # add a (trainable) layer to the NN
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        # Call next-method to get an input tensor and a label tensor from the data layer
        # Store label tensor for backward pass
        input_array, self.label_tensor = self.data_layer.next()
        out_arr, regularization = self.raw_forward_pass(input_array)
        return self.loss_layer.forward(out_arr, self.label_tensor), regularization

    def raw_forward_pass(self, input_array):
        # Perform raw forward pass calculating reg loss
        out_arr = input_array
        regularizer_loss = 0
        for layer in self.layers:
            regularizer_loss += layer.calculate_regularization_loss()
            out_arr = layer.forward(out_arr)
        return out_arr, regularizer_loss

    def backward(self):
        # Backward pass the loss through the NN
        backward_array = self.loss_layer.backward(self.label_tensor)
        # Reverse list with NN layers -> backward pass
        for layer in reversed(self.layers):
            backward_array = layer.backward(backward_array)

    def train(self, iterations):
        # NN training
        self.phase = "Train"
        for layer in self.layers:
            layer.testing_phase = False
        # 1. forward pass data through NN and return loss
        # 2. store loss for visualization of the loss development
        # 3. backward pass loss through reversed NN to update the weights
      
        iter = 0
        while iter < iterations:
            iter += 1
            loss, regularization_loss = self.forward()
            self.loss.append(loss + regularization_loss)
            self.backward()

    def test(self, input_array):
        # Forward pass given input tensor through the network
        self.phase = "Test"
        for layer in self.layers:
            layer.testing_phase = True
        resulting_array, _ = self.raw_forward_pass(input_array)
        # Last iteration delivers the loss -> return loss
        return resulting_array
