# !/usr/bin/env python 

import argparse
import pandas as pd
import numpy as np
import math
from neuralClasses import NeuronalNetwork, Layer
from preprocessing import preprocessing


# In -> h1 -> h2 -> Out

#ReLu 
#derivee output/input
def relu_derivative(X):
    return 1 if X > 0 else 0

# def mse_derivative(y_pred, Y, X):
    # return np.mean((y_pred - Y) * X, axis = 0)



def init_network(my_network, data, show_network=0):
    # Build chosen number of layers / neurons
    my_network.layers.append(Layer(data.nb_features, int(math.log(data.nb_features) ** 2)))
    # my_network.layers.append(Layer(data.nb_features, int(math.log(data.nb_features) ** 2)))
    while int(math.log(my_network.layers[-1].n_neurons) ** 2) > data.nb_output:
        my_network.layers.append(Layer(my_network.layers[-1].n_neurons, int(math.log(my_network.layers[-1].n_neurons) ** 2)))
    my_network.layers.append(Layer(my_network.layers[-1].n_neurons, data.nb_output))
    layers = [layer.n_neurons for layer in my_network.layers]

    # Initialize first layer with X datas 
    for l, layer in enumerate(my_network.layers):
        for n, neuron in enumerate(layer.neurons):
            if l == 0:
                neuron.compute_neuron_output(data.X[:, :])
            else:
                neuron.compute_neuron_output(my_network.layers[l - 1].layer_output)
        layer.get_layer_output()

    if show_network:
        print("\033[1m" + "\nNetwork : " + '\033[0m')
        print("Nombre de layers :", len(layers))
        print("Nombre de neurones / layer :", layers)
        for layer in my_network.layers:
            print(np.mean(layer.layer_output, axis = 0))
        # exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv datas file")
    parser.add_argument("-p", "--preprocess", 
        help="Show preprocessed data",
        action="store_true")
    parser.add_argument("-n", "--network", 
        help="Show network data",
        action="store_true")
    args = parser.parse_args()
    file_name = args.file


    data = preprocessing(file_name, args.preprocess)

    my_network = NeuronalNetwork(data.X, data.Y)
    init_network(my_network, data, args.network)
    # exit()

    my_network.fit()


if __name__ == "__main__":
    main()