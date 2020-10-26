import numpy as np
import random
import math

class NeuronalNetwork:
    def __init__(self, X, Y, cycles=1000, alpha=0.05):
        self.layers = []
        self.cycles = cycles
        self.alpha = alpha
        self.X = X
        self.Y = np.array(Y)
        self.cost = 1

    #derivee cost par rapport au poid
    def compute_weight_derivative(self, neuron, Y, X):
        if neuron.neuron_output <= 0:
            result = 0
        else:
        # print(np.shape(y_pred))
            result = 2 * np.mean(np.dot(neuron.neuron_output - Y[:, 0], neuron.inputs.transpose()), axis = 0)
        return result

    def compute_bias_derivative(self, neuron, Y):
        # if y_pred <= 0:
        #     result = 0
        # else:
        result = 2 * np.mean(neuron.neuron_output - Y[:, 0], axis = 0)
        return result

    def _cost(self):
        result = np.mean((self.Y - self.predict) ** 2, axis = 0)
        self.cost = np.sum(result)

    def _predict(self):
        self.predict = self.layers[-1].layer_output

    def fit_forward(self):
        for l, layer in enumerate(self.layers):
            if l == 0:
                continue
            for neuron in layer.neurons:
                neuron.compute_neuron_output(self.layers[l - 1].layer_output)

    def backpropagation(self):
        l = len(self.layers) - 1
        while l:
            for neuron in self.layers[l].neurons:
                neuron.weights = neuron.weights - self.alpha * self.compute_weight_derivative(neuron, self.Y, self.X)
                neuron.bias = neuron.bias - self.alpha * self.compute_bias_derivative(neuron, self.Y)
                print("neuron weights : ", neuron.weights)
            l -= 1

    def fit(self):
        i = 0
        while i < 10:
            self.fit_forward()
            self._predict()
            print("Predicted output : ", self.predict)
            self._cost()
            print("Cost : ", self.cost)
            # exit()
            self.backpropagation()
            print(i)
            i+= 1

class Layer(NeuronalNetwork):
    def __init__(self, inputs, n_neurons):
        self.neurons = []
        self.n_neurons = n_neurons

        for i in range(self.n_neurons):
            self.neurons.append(Neuron(inputs))
    
    def get_layer_output(self):
        # self.layer_output = []
        self.layer_output = [neuron.neuron_output for neuron in self.neurons]
        self.layer_output = np.transpose(self.layer_output)
        # print("\nLayer output : ", self.layer_output)

class Neuron(Layer):
    def __init__(self, inputs):
        self.weights = np.zeros(inputs)
        # self.weights = np.random.randn(inputs)
        self.bias = 0
    
    def compute_neuron_output(self, inputs):
        # print("inputs : \n", np.shape(inputs))
        # print("self.weights : ", np.shape(self.weights))
        self.inputs = inputs
        self.neuron_output = np.sum(inputs * self.weights, axis = 1)  + self.bias        
        self.neuron_output = np.maximum(0, self.neuron_output)
        # print("shape output : \n", np.shape(self.neuron_output))