#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[14]:


import numpy
import scipy.special


# # Creating a neural network class

# In[15]:


class neuralNetwork():
   
    # Initialising neural_network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Setting the number of input, hidden and output nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Setting Learning Rate
        self.lr = learningrate
        
        # Creating initial weights
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # Activating function for summed signals 
        self.activation_function = lambda x: scipy.special.expit(x)
          
    
    # Method for training a network
    def train(self, inputs_list, targets_list):
        
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Activates calculated signals
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Activates calculated signals
        final_outputs = self.activation_function(final_inputs)
        
        # Finding output error
        output_errors = targets - final_outputs
        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # Update the weights for the links between the hidden and output layers with backpropagation
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers with backpropagation
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    
    # Method for quering netwrk
    def query(self, inputs_list):
        # Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Activates calculated signals
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into output layer       
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Activates calculated signals
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    

