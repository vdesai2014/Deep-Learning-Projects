import numpy as np
import matplotlib.pyplot as plt
import math 
import h5py
import time
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

np.random.seed(1)

class NeuNetVisualization:
    def __init__(self, layer_dims, max_weight=0.6):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 10)) 
        self.ax.axis('off')
        self.coordinates =[]
        self.neurons = []
        self.synapses = []
        self.max_weight = max_weight
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
                    synapse = Line2D([previous_neuron_x, current_neuron_x], [previous_neuron_y, current_neuron_y], linewidth=0.15, color = 'w')
                    self.ax.add_line(synapse)
                    neuron_synapses.append(synapse)
                layer_synapses.append(neuron_synapses)
            self.synapses.append(layer_synapses)

    def update(self, parameters, activations):
        for i in range(len(parameters)):
            row, col = parameters[i].shape 
            for j in range(row):
                for k in range(col):
                    new_synapse_strength= 2 * (parameters[i][j][k]/self.max_weight)
                    self.synapses[i][j][k].set_linewidth(new_synapse_strength)
        num_updates = activations[0].shape[1]
        for sample in range(num_updates):
            for layer in range(len(self.neurons)):
                for neuron in range(len(self.neurons[layer])):
                    neuron_activation = activations[layer][neuron][sample]
                    if layer < len(self.neurons)-1 and neuron_activation > 0.01:
                        self.neurons[layer][neuron].set_facecolor([1, 0, 0])
                    elif layer == len(self.neurons)-1 and neuron_activation > 0.5:
                        self.neurons[layer][neuron].set_facecolor([1, 0, 0])
            self.current_frame += 1
            self.fig.canvas.draw()

            plt.savefig("Frame - " + str(self.current_frame) + ".png")
            self.resetActivations() 
            
    
    def resetActivations(self):
        for layer in range(len(self.neurons)):
            for neuron in range(len(self.neurons[layer])):
                self.neurons[layer][neuron].set_facecolor('b')

def initialize_parameters(layerdims):
    '''layerdims: List containing no. of units in each layer
    
    Returns:
    parameters: A dict consist of all learnable parameters (W1,b1, W2,b2, ...)
    '''
    parameters={}
    L = len(layerdims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn( layerdims[i], layerdims[i-1]) * 0.1
        parameters["b"+str(i)] = np.zeros( (layerdims[i],1))
    
        
    return parameters

def forward(A_prev, W, b, activation):
    ''' Forward Propagation for Single layer
    A_prev: Activation from previous layer (size of previous layer, Batch_size)
        A[0] = X
    W: Weight matrix (size of current layer, size of previous layer)
    b: bias vector, (size of current layer, 1)
    
    Returns:
    A: Output of Single layer
    cache = (A_prev,W,b,Z), these will be used while backpropagation
    '''
    global biggestparameter 
    biggestparameter = max(np.max(W), biggestparameter)

    # Linear 
    Z = np.dot(W,A_prev) + b
    # Activation Function 
    if activation== "sigmoid":
        A=1/(1+np.exp(-Z))
        
    if activation== "relu":
        A = np.maximum(0,Z)
    cache=(A_prev,W,b,Z)
    
    return A, cache 

def L_layer_forward(X, parameters, layerdims):
    ''' Forward propagation for L-layer
     
     [LINEAR -> RELU]*(L-1)   ->    LINEAR->SIGMOID
     
     X: Input matrix (input size/no. of features, no. of examples/BatchSize)
     parameters: dict of {W1,b1 ,W2,b2, ...}
     layerdims: Vector, no. of units in each layer  (no. of layers,)
    
    Returns:
     y_hat: Output of Forward Propagation
     caches: (A_prev,W,b,Z) *(L-1 times , of 1,2,..L layers)
    '''
    caches=[]
    L =  len(layerdims)-1
    A = X
    
    # L[0] is units for Input layer
    # [LINEAR -> RELU]*(L-1)    Forward for L-1 layers 
    for l in range(1,L):
        A_prev = A
        A, cache=forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
      
    # Forward for Last layer
    # [Linear -> sigmoid]
    y_hat, cache=forward(A, parameters["W"+str(l+1)], parameters["b"+str(l+1)], "sigmoid")
    caches.append(cache)
    
    return y_hat, caches
    
def compute_cost(y_hat, Y):
    '''Computes the Loss between predicted and true label
    y_hat: Predicted Output (1, no. of examples)
    Y: Actual label vector consist of 0/1 (1, no. of examples)
    
    '''
    m = Y.shape[1]
    #costt = (Y*np.log(y_hat)) + ((1-Y)*np.log(1-y_hat))
    costt = np.add( np.multiply(Y, np.log(y_hat)) ,  np.multiply(1-Y, np.log(1-y_hat))  ) 
    cost = (-1/m) * np.sum(costt, axis=1)
    return cost

def backward(dA,  cache, activation):
    '''Backward propagation for single layer
    dz: Derivative of Cost wrt Z   (of current layer)
    cache: tuple of (A_prev,W,b,Z)
    
    Retuns:
    dW: Gradient of Cost wrt W, (having same shape as W)
    db: Gradient of Cost wrt b, (having same shape as b)
    dA_prev: Gradient of Cost wrt Activation (same shape as A_prev)
    '''
    A_prev,W,b,Z = cache
    m= A_prev.shape[1]
    
    # Computing derivative of Cost wrt Z
    # dA, Z, 
    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <=0] =0
        
    if activation == "sigmoid":
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        
    # Computing derivative of Cost wrt A & W  & b
    dA_prev = np.dot(W.transpose(), dZ)
    dW = (1/m) * np.dot(dZ, A_prev.transpose())
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    return dA_prev, dW, db

def L_layer_backward(y_hat, Y, caches, layerdims):
    ''' Backward Propogation from layer L to 1
    y_hat:  predicted output
    Y:      true values
    caches: list of caches stored while forward Propagation
                (A_prev,W,b,Z) *(L-1 times , of 1,2,..L-1 layers) with relu
                (A_prev,W,b,Z) (for layer L, with sigmoid)
    layerdims:List having no. of units in each layer 
    Returns:
    grads: A dict containing gradient (dA_i, dW_i, db_i), this will be used while updating parameters
    '''
    AL = y_hat 
    L = len(layerdims) -1
    grads={}
    
    # Intializing Backpropagation
    
    # Compute derivation of Cost wrt A
    dAL = -(Y/AL- ((1-Y)/(1-AL)))

    # Compute derivative of Lth layer (Sigmoid -> Linear) gradients. 
    # Inputs: (AL, Y, caches)     Outputs: (grad["dAL"], grad["dWL] , grad["dbL"])
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = backward(dAL, caches[-1], activation="sigmoid")
    
   
    # Compute derivative for (1,2,..L-1)layers (relu -> Linear) gradients.
    # Inputs:(grads[dAL], caches)   Outputs:(grads(dA_i, dW_i, db_i)
   
    for i in list(reversed(range(L-1))):                       
        
        current_cache = caches[i]
        
        a,b,c=backward(grads["dA"+str(i+2)], current_cache, activation="relu")
        grads["dA"+str(i+1)] = a
        grads["dW"+str(i+1)] = b
        grads["db"+str(i+1)] = c
        
    return grads

def update_params(params, grads, learning_rate):
    '''
    parameters: dict of (W1,b1, W2,b2,...)
    grads: Gradients of(A,W,b) stored while Backpropagation (dA,dW,db)
    
    returns: updated parameters
    '''
    # As each layer has 2 parameters (W,b)
    L=len(params) // 2
    
    for l in range(L):
        params["W"+str(l+1)] = params["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        params["b"+str(l+1)] = params["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]
        
    return params

def load_data():
    train_dataset = h5py.File('/home/vrushank/Documents/GitHub/Deep-Learning-Projects/Neural Network Visualizer/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading DataSet
train_x, train_Y, test_x, test_Y, classes = load_data()
print(f"Training Data{train_x.shape}")
print(f"Test data{test_x.shape}")

# Reshape each image to Vector, X to (Single_Vector 64*64*3,  no.of examples)
train_Xflat = train_x.reshape(-1, train_x.shape[0])
test_Xflat = test_x.reshape(-1, test_x.shape[0])

# Scaling pixel values b/w 0 to 1 
train_X = train_Xflat /255
test_X = test_Xflat /255

# Model Configuration
# len(layer_dims), will be no. of layers with Input & Output layers
layer_dims=[12288, 20, 7, 5, 1]

learning_rate=0.009

# No. of Gradient Descent Iterations

def image_classifier(X, Y, layer_dims, learning_rate, num_itr, parameters, initialize=False):
    ''' Implements a L-layer NN: 
    [Linear->Relu] *(L-1)times  ->  [Linear->Sigmoid] 
    
    X: Input data(Images) (Height*Weidth*3 , no. of examples)
    Y: True labels, consist of 0/1      (1, no. of examples)
    layer_dims: list, where each value is no. of units.
    learning_rate: for parameters update
    num_itr: no. of iterations of Gradient Descent
    
    Returs:
    parameters: parameters learnt during Model Training.
    '''
    frames_per_second = 30
    gif_length = 30
    total_frames = frames_per_second*gif_length
    interval_gap = 10 
    num_saved_iterations = num_itr/interval_gap
    samples_to_save = int(total_frames//num_saved_iterations)

    costs=[]
    
    if initialize:
        parameters = initialize_parameters(layer_dims)
        initial_parameters = []
        for l in range(1, len(layer_dims)-1):
            initial_parameters.append(parameters["W"+str(l+1)]) 
    
    neu_net_visualizer = NeuNetVisualization(layer_dims[1:])
    # Gradient Descent
    for i in range(num_itr):
        
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        y_hat, caches = L_layer_forward(X, parameters, layer_dims)


        # Compute Cost
        cost = compute_cost(y_hat, Y)

        # Backward propagation
        grads = L_layer_backward(y_hat, Y, caches, layer_dims)

        # Update Parameters
        parameters = update_params(parameters, grads, learning_rate)
        new_parameters = []
        new_activations = []
        for layer in range(1, len(layer_dims)-1):
                new_activations.append(caches[layer][0][:, :samples_to_save])
                new_parameters.append(caches[layer][1])
        
        if i%interval_gap == 0:
            new_parameters = []
            new_activations = []
            for layer in range(1, len(layer_dims)-1):
                new_activations.append(caches[layer][0][:, :samples_to_save])
                new_parameters.append(caches[layer][1])
            new_activations.append(y_hat[:, :samples_to_save])
            for idx, j in enumerate(initial_parameters):
                new_parameters[idx] = np.maximum((new_parameters[idx] - j), 0)
            neu_net_visualizer.update(new_parameters, new_activations)
            

        if i%200 ==0:
            print(f"cost {i}: {np.squeeze(cost)}")
        if i%100 ==0:
            costs.append(cost)
        if i == 800:
            print(y_hat[0, :10])
            print()
            print(Y[0, :10])

    # Ploting the Cost
    plt.plot(costs)
    plt.xlabel("n iteration")
    plt.ylabel("cost")
    return parameters

biggestparameter = 0
parameters=image_classifier(train_X, train_Y, layer_dims, 0.05, 2000, 0, initialize=True)