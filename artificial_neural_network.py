#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:58:16 2017

@author: Jean Silva <me@jeancsil.com>
"""

import numpy as np
                           
class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        
        # Single neuron with 3 input connections and 1 output connection.
        # Random weights assigned to a 3 x 1 matrix, with values in the range -1 to 1 and mean 0.
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
                                                    
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in iter(range(number_of_training_iterations)):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output
            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # Adjust the weights.
            self.synaptic_weights += adjustment
            if (iteration % 1000 == 0):
                print ("error after %s iterations: %s" % (iteration, str(np.mean(np.abs(error)))))

    # The neural network 'thinks'
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))
    
if __name__ == "__main__":
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 0]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T
                              
    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Test the neural network with a new pattern
    test = [1, 0, 0]
    print ("Considering new situation %s -> ?: " % test )
    print (neural_network.think(np.array(test)))