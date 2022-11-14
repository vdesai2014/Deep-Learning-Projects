from functools import cache
from mimetypes import init
import matplotlib.pyplot as plt
import numpy as np
import h5py
import math 
import imageio

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True); #why is the derivation of relu just Z? should be equal to 1???
    dZ[Z <= 0] = 0 

    return dZ

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 2/math.sqrt(layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_activation_forward(A_prev, W, b, activation):

    linear_cache = A_prev, W, b
    Z = np.dot(W, A_prev) + b

    if activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))

    elif activation == "relu":
        A = np.maximum(0,Z)

    cache = (linear_cache, Z)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A 
        A, linear_activation_cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(linear_activation_cache)

    AL, linear_activation_cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(linear_activation_cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost
    
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    dA_prev, dW, db = linear_activation_backward(dAL, caches[L - 1], "sigmoid")
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev, dW, db

    for l in reversed(range(L-1)):
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], caches[l], "relu")
        grads["dA" + str(l + 1)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.00825, num_iterations = 200, print_cost=False, batchSize=1):
    np.random.seed(69)
    costs = []                        
    parameters = initialize_parameters_deep(layers_dims)

    fps = 1
    gifLength = 60 
    iterationInterval = round(num_iterations/(fps*gifLength))
    frameCount = 0 
    
    for i in range(0, num_iterations):
        completionState = i/num_iterations
        AL, caches =L_model_forward(X, parameters)
        if(i % iterationInterval == 0 or i == num_iterations):
            activations, predictionResults = identifyActivations(caches, AL, batchSize, train_yx, i)
            for i in range(batchSize):
                frameCount += 1
                visualizeForwardPass(activations[i], layers_dims, completionState, predictionResults[i], frameCount)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    createGif(frameCount)

    return parameters

def visualizeForwardPass(activationList, listdims, completionState, predictionResult, frameCount): 
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10)) 
    ax.axis('off')

    padding = 0.1
    padSpacing = 0.8/len(listdims)
    master_coordinate_list = []

    for i in range(len(listdims)):
        layer_y = padding + (padSpacing*i)
        if (listdims[i] == 1):
            coordinate_list = []
            neuron_x = 0.5
            coordinate_list.append([neuron_x, layer_y])
        else: 
            coordinate_list = []
            x_spacing = (1-padding*2)/(listdims[i])
            x_origin = 0.5 - ((listdims[i]-1)/2)*x_spacing      
            for j in range(listdims[i]):
                    neuron_x = x_origin + (x_spacing)*j
                    coordinate_list.append([neuron_x, layer_y])   
        master_coordinate_list.append(coordinate_list)

    for i in range(len(master_coordinate_list)):
        for j in range(len(master_coordinate_list[i])):
            circle = plt.Circle((master_coordinate_list[i][j][0], master_coordinate_list[i][j][1]), 0.005, color='r', zorder=10)
            ax.add_patch(circle)

    for i in range(len(master_coordinate_list)-1):
        for j in range(len(master_coordinate_list[i])):
            for k in range(len(master_coordinate_list[i+1])):
                plt.plot([master_coordinate_list[i][j][0], master_coordinate_list[i+1][k][0]], [master_coordinate_list[i][j][1], 
                master_coordinate_list[i+1][k][1]], 'b')
                
    for i in reversed(range(1, len(activationList))):
        for j in (range(len(activationList[i]))):
            if activationList[i][j] > 0:
                for k in range(len(master_coordinate_list[i-1])):
                    plt.plot([master_coordinate_list[i][j][0], master_coordinate_list[i-1][k][0]], [master_coordinate_list[i][j][1], 
                master_coordinate_list[i-1][k][1]], color = 'w', linewidth = activationList[i][j]*3)

    progressBar = plt.Rectangle((0.15,0.8), 0.2*completionState, 0.03, color = 'w')
    ax.add_patch(progressBar) 
    plt.text(0.15, 0.77, str("Percent Completed: " + str(round(completionState*100, 2)) + "%"))
    plt.text(0.295, 0.07, str(math.ceil(activationList[0][0])), fontsize = 10)
    plt.text(0.695, 0.07, str(math.ceil(activationList[0][1])), fontsize = 10)
    if (predictionResult == 1):
        plt.text(0.725, 0.77, str("Correct"), color = 'g', fontsize = 30)
    else: 
        plt.text(0.725, 0.77, str("Incorrect"), color = 'r', fontsize = 30)
    outputResult = activationList[5][0] #THIS LINE IS HARD CODED, UPDATE LATER
    plt.text(0.495, 0.78, str(outputResult), color = 'w', fontsize = 10)
    plt.savefig(f'./img/' + str(frameCount) + '.png')
    plt.close()
    
def identifyActivations(caches, AL, batchSize, expectedOutput, numIterations):
    activationList = []
    masterActivationList = []
    for z in range(batchSize):
        batchActivationList = []
        for i in range(len(caches)):
            layer_activation_list = []
            for j in range(len(caches[i][0][0])):
                if(caches[i][0][0][j][z] > 0.01):
                      activationColor = (caches[i][0][0][j][z] / 30)
                      layer_activation_list.append(activationColor)
                else:
                      layer_activation_list.append(0)
            batchActivationList.append(layer_activation_list)
        if(AL[0][z] > 0.11):
            batchActivationList.append([1])
        else:
            batchActivationList.append([0])
        masterActivationList.append(batchActivationList)

    predictionResult = []
    for i in range(len(AL[0])):
        if ((AL[0][i] > 0.95 and expectedOutput[i] == 1) or (AL[0][i] < 0.11 and expectedOutput[i] == 0)):
            predictionResult.append(1)
        else:
            predictionResult.append(0)

    return masterActivationList, predictionResult

def createGif(frameCount):
    frames = []

    for i in range(1, frameCount):
        image = imageio.imread(f'./img/' + str(i) + '.png')
        frames.append(image)
    
    imageio.mimsave('./example.gif', frames, fps = 5)

inputList = [[1,1], [0, 1], [1, 0], [0, 0]]
array = np.array(inputList) 
train_x = array.reshape(array.shape[0], -1).T
train_yx = [0, 1, 1, 0]
train_y = np.array(train_yx, ndmin = 2)
layers_dims = [2, 5, 5, 5, 5, 1]
costs = []

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 20000, print_cost = True, batchSize = 4)
