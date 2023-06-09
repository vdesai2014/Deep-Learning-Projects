import numpy as np
import matplotlib.pyplot as plt
import math 
import h5py
import time
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

np.random.seed(1)

class NeuNetVisualization:
    def __init__(self, layer_dims, max_weight, max_activation):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 10)) 
        self.ax.axis('off')
        self.coordinates =[]
        self.neurons = []
        self.synapses = []
        self.max_weight = max_weight
        self.max_activation = max_activation
        self.current_frame = 0 

        x_extent = 1
        y_extent = 1 
        y_spacing = y_extent/(len(layer_dims))
        y_origin = y_spacing/2
        neuron_radius = 0.01
        
        for i in range(len(layer_dims)):
            layer_coordinates = []
            layer_y = y_origin + i*y_spacing
            layer_neurons = []
            x_spacing = x_extent/layer_dims[i]
            x_origin = x_spacing/2
            for j in range(layer_dims[i]):
                neuron_x = x_origin + j*x_spacing
                neuron_coordinates = (neuron_x, layer_y)
                neuron = Circle(neuron_coordinates, radius = neuron_radius, zorder = 10, color = 'w')
                layer_coordinates.append(neuron_coordinates)
                layer_neurons.append(neuron) 
                self.ax.add_patch(neuron)
            self.neurons.append(layer_neurons)
            self.coordinates.append(layer_coordinates)
    
        for i in range(1, len(self.coordinates)):
            layer_synapses = []
            for j in range(len(self.coordinates[i])): #loops 7 times
                neuron_synapses = []
                current_neuron_x, current_neuron_y  = self.coordinates[i][j]
                for k in range(len(self.coordinates[i-1])): #loops 20 times 
                    previous_neuron_x, previous_neuron_y = self.coordinates[i-1][k]
                    synapse = Line2D([previous_neuron_x, current_neuron_x], [previous_neuron_y, current_neuron_y], linewidth=1, color = 'w')
                    self.ax.add_line(synapse)
                    neuron_synapses.append(synapse)
                layer_synapses.append(neuron_synapses)
            self.synapses.append(layer_synapses)

    def update(self, parameters, activations):
        for i in range(len(parameters)):
            row, col = parameters[i].shape 
            for j in range(row):
                for k in range(col):
                    new_line_width = 10 * (parameters[i][j][k]/self.max_weight)
                    self.synapses[i][j][k].set_linewidth(new_line_width)
        num_updates = activations[0].shape[1]
        print(num_updates)
        for sample in range(num_updates):
            for layer in range(len(self.neurons)):
                for neuron in range(len(self.neurons[layer])):
                    new_neuron_activation = (activations[layer][neuron][sample]/self.max_activation)
                    if new_neuron_activation > 0.01:
                        self.neurons[layer][neuron].set_facecolor([new_neuron_activation, 0, 0])
            self.current_frame += 1
            self.fig.canvas.draw()
            plt.savefig("Frame Number: " + str(self.current_frame) + ".png")
            self.resetActivations() 
            
    
    def resetActivations(self):
        for layer in range(len(self.neurons)):
            for neuron in range(len(self.neurons[layer])):
                self.neurons[layer][neuron].set_facecolor('b')
        
initialize = NeuNetVisualization([1, 2], 5, 5)
W1 = np.array([[1]])
W2 = np.array([[1], [1]])
initialize.update([], [W1, W2])
W1 = np.array([[2]])
W2 = np.array([[1], [1]])
initialize.update([], [W1, W2])
W1 = np.array([[0]])
W2 = np.array([[1], [1]])
initialize.update([], [W1, W2])

