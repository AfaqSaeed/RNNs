import numpy as np
from .Base import BaseLayer
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # initialize the sizes of the input, hidden and output layers
        self.trainable = True
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.concat_size = input_size + hidden_size
        
        # initialize properties
        self.optimizer = None
        self._gradient_weights = None
        self._memorize = False

        # initialize the hidden state
        self.hidden_state = np.zeros((self.hidden_size,))

        # initialize the layers of the Elman RNN
        # Fully connected layer to calculate the hidden state
        self.fc1_hidden_state = FullyConnected(self.concat_size, hidden_size)
        # Fully connected layer to calculate the output of the cell
        self.fc2_output = FullyConnected(hidden_size, output_size)
        # tanh for squashing the hidden state to values from -1 to 1
        self.tanh_activation = TanH()
        # sigmoid for squashing the output from 0 to 1
        self.sigmoid_activation = Sigmoid()

    def forward(self, input_tensor):

        # retrieve the sequence length
        self.sequence_length = input_tensor.shape[0]

        # if the memory flag is set to zero, reset the hidden state 
        if not self._memorize:
            self.hidden_state = np.zeros((self.hidden_size,))

        # store the input tensor for the backward pass
        self.input_tensor = input_tensor
        # initialize the memory  
        # each row represents the corresponding values at a time step
        self.hidden_state_memory = np.zeros((self.sequence_length, self.hidden_size))
        self.output_memory = np.zeros((self.sequence_length, self.output_size))
        self.fc1_output_memory = np.zeros((self.sequence_length, self.hidden_size))
        self.fc1_input_memory = np.zeros((self.sequence_length, self.concat_size+1))
        self.fc2_output_memory = np.zeros((self.sequence_length, self.output_size))
        self.fc2_input_memory = np.zeros((self.sequence_length, self.hidden_size+1))
        
        for t in range(self.sequence_length):
            # concatenate the input vector and the hidden state in a current vector
            self.input_forward_pass = np.concatenate((input_tensor[t], self.hidden_state))
             # Add new axis to input tensor n
            self.fc1_input = np.expand_dims(self.input_forward_pass, 1).T
            # calculate the input for the tanh activation function
            self.fc1_output = self.fc1_hidden_state.forward(self.fc1_input)
            self.fc1_output_memory[t] = self.fc1_output
            self.fc1_input_memory[t] = self.fc1_hidden_state.X
            # calculate the hidden state
            self.hidden_state = self.tanh_activation.forward(self.fc1_output)
            self.hidden_state_memory[t] = self.hidden_state
            # calculate the input for the sigmoid activation function
            self.fc2_output_values = self.fc2_output.forward(self.hidden_state)
            self.fc2_output_memory[t] = self.fc2_output_values
            self.fc2_input_memory[t] = self.fc2_output.X
            # calculate the output of the cell 
            self.output = self.sigmoid_activation.forward(self.fc2_output_values)
            self.output_memory[t] = self.output
            # remove additional dimension of hidden state
            self.hidden_state = np.squeeze(self.hidden_state, axis=0)
            
        return self.output_memory
    
    def backward(self, error_tensor):
        '''
        Method to backpropagate the error tensor through the Elman RNN
        '''
        sequence_length = error_tensor.shape[0]
        # Check memorize flag 
        if not self._memorize:
            self.hidden_state_gradient = np.zeros(self.hidden_size)

        fc1_hidden_state_gradient_sum = np.zeros_like(self.fc1_hidden_state.weights)
        error_tensor_backward = np.zeros((sequence_length, self.input_size))
        
        for t in reversed(range(sequence_length)):
            # load the saved inputs and activations from the memory of the forward pass 
            #self.fc1_hidden_state.X = self.fc1_input_memory[t]
            #self.fc2_output.X = self.fc2_input_memory[t]
            self.sigmoid_activation.X = self.output_memory[t]
            self.tanh_activation.X = self.hidden_state_memory[t]

            # load the saved inputs and activations from the memory of the forward pass 
            self.fc1_hidden_state.X = np.expand_dims(self.fc1_input_memory[t],1).T
            self.fc2_output.X = np.expand_dims(self.fc2_input_memory[t],1).T
            #self.sigmoid_activation.X = np.expand_dims(self.fc2_output_memory[t],1)
            #self.tanh_activation.X = np.expand_dims(self.hidden_state_memory[t],1)

            # calculate the gradients for the output fully connected layer and the sigmoid activation
            sigmoid_gradient = self.sigmoid_activation.backward(error_tensor[t])
            fc2_output_gradient = self.fc2_output.backward(np.expand_dims(sigmoid_gradient,1).T)

            # for the first iteration of the BPTT the hidden state gradient is zero
            if t == sequence_length - 1:
                self.hidden_state_gradient = np.zeros(self.hidden_size)
            # calculate the error sum of the hidden state and the output fully connected layer
            error_sum = self.hidden_state_gradient + fc2_output_gradient
            # calculate the gradient for the tanh activation function
            tanh_gradient = self.tanh_activation.backward(error_sum)
            # calculate the gradient for the input fully connected layer
            fc1_hidden_state_gradient = self.fc1_hidden_state.backward(tanh_gradient)

            # build a sum of the gradients for the input fully connected layer
            # necessary for the update of the weights
            fc1_hidden_state_gradient_sum += self.fc1_hidden_state.gradient_weights

            # extract the gradient for the hidden state for the next iteration
            fc1_hidden_state_gradient = np.squeeze(fc1_hidden_state_gradient)
            self.hidden_state_gradient = fc1_hidden_state_gradient[self.input_size:]
            # extract the gradient for the input for the current iteration
            error_tensor_backward[t] += fc1_hidden_state_gradient[:self.input_size]
        
        self._gradient_weights = fc1_hidden_state_gradient_sum
        # zero_row = np.zeros((1, self.hidden_size))
        # fc1_hidden_state_gradient_sum = np.concatenate((fc1_hidden_state_gradient_sum, zero_row), axis=0)
        # Update the weights using the sum of the gradients
        if self.optimizer is not None:
            self.fc1_hidden_state.weights = self.optimizer.calculate_update(self.fc1_hidden_state.weights, fc1_hidden_state_gradient_sum)
        
        return error_tensor_backward
    
    def initialize(self, weights_initializer, bias_initializer):
        '''
        Method to initialize the weights and bias of the fully connected layers of the Elman RNN
        '''
        self.fc1_hidden_state.initialize(weights_initializer, bias_initializer)
        self.fc2_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        '''
        Method to calculate the regularization loss
        '''
        regularization_loss = 0
        if self.optimizer is not None:
            regularization_loss = self.optimizer.calculate_regularization_loss(self.fc1_hidden_state.weights)
            regularization_loss += self.fc2_output.calculate_regularization_loss()
        return regularization_loss
        
    # getter method for optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    # setter method for optimizer property
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    @property
    def memorize(self):
        return self._memorize
    # setter method for optimizer property
    @memorize.setter
    def memorize(self, memorize_flag):
        self._memorize = memorize_flag

    # getter and setter method for weights property
    @property
    def weights(self):
        return self.fc1_hidden_state.weights
    @weights.setter
    def weights(self, weights):
        self.fc1_hidden_state.weights = weights

    # getter method for gradient weights property
    @property
    def gradient_weights(self):
        # Return the gradients with respect to the weights
        return self._gradient_weights

