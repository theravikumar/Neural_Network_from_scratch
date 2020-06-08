#!/usr/bin/env python
# coding: utf-8
# Author: Ravi Kumar 
# <ravi940107@gmail.com>
#




get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

#read csv file
no_of_different_labels = 10
train_data = np.loadtxt("mnist_train.csv",delimiter=",")
test_data = np.loadtxt( "mnist_test.csv", delimiter=",") 

#Normalization of image value from (0,255) to (0,1)
fac = 0.99 / 255

# we add 0.01 because we don't want zero
# Removes first column as it contain label
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01 
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

# select first column as it contain labels
train_labels = np.asfarray(train_data[:, :1]) 
test_labels = np.asfarray(test_data[:, :1])


lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# creates the layers, apply forward propagation and activation  
class Neural_Layers:
    def __init__(self,n_InNeurons,n_Neurons):
        self.weights = np.random.randn(n_InNeurons,n_Neurons)
        self.bias = np.random.randn(n_Neurons)
        
    def Activation_Func(self,out):
        acti = 1/(1+np.exp(-out))
        return (acti)
        
    def forward(self,x):
        outputs = np.dot(x,self.weights) + self.bias
        self.activated = self.Activation_Func(outputs)
        return (self.activated)
    def apply_activation_derivative(self, r):
        return r * (1 - r)

        
# Class to test and train Neural Network    
class Neural_Network:
    def __init__(self):
        self.Layers_List = []
        
    def Add_Layer(self,Layer):
        self.Layers_List.append(Layer)
        
    def Feed_Forward(self,inputs):
        for layer in self.Layers_List:
            inputs = layer.forward(inputs)
#             print(inputs)
        return (inputs)
            
    def predict(self, X):

        ff = self.Feed_Forward(X)
        ff=np.atleast_2d(ff)

        return((ff == ff.max(axis=1)[:,None]).astype(int))


    def Backpropagation(self,x,y,Learn_Rate):
        output = self.Feed_Forward(x)

        for i in reversed(range(len(self.Layers_List))):
            layer = self.Layers_List[i]

            if layer == self.Layers_List[-1]:
                layer.error = y - output
                
                # The output = layer.activated in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)

            else:
                next_layer = self.Layers_List[i + 1]
                
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.activated)


        # Update the weights
        for i in range(len(self.Layers_List)):
            layer = self.Layers_List[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(x if i == 0 else self.Layers_List[i - 1].activated)
            layer.weights += layer.delta * input_to_use.T * Learn_Rate
    def train(self, X, y, Learn_Rate, max_epochs):


        mses = []

        for i in range(max_epochs):
            for j in range(len(X)):
                self.Backpropagation(X[j], y[j], Learn_Rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y - nn.Feed_Forward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses

x=train_imgs
y=train_labels_one_hot

#Make neural network
nn=Neural_Network()
nn.Add_Layer(Neural_Layers(784,20))
nn.Add_Layer(Neural_Layers(20,20))
nn.Add_Layer(Neural_Layers(20,10))


errors = nn.train(x, y, 0.3, 1000)

# Find the Accuracy
counter = 0
l1=((test_labels_one_hot == test_labels_one_hot.max(axis=1)[:,None]).astype(int))
l2=nn.predict(test_imgs)

for i in range (len(l1)):
    if list(l1[i])==list(l2[i]):
        counter += 1
print("The Accuracy is : "(counter/len(l1))*100)

