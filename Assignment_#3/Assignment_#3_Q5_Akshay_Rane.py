#!/usr/bin/env python
# coding: utf-8

# In[123]:


import _pickle as cPickle
import gzip
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# plot charts inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


#Load the dataset
def separate_componenets_frame(dataframe_set):
  
    X_set,Y_set = dataframe_set[0],dataframe_set[1]
    X_set = X_set / 255
    
    y_set_normalize = np.zeros(Y_set.shape)
    y_set_normalize[np.where(Y_set == 0.0)[0]] = 1
    Y_set = y_set_normalize
    return X_set,Y_set
    
def sigmoid_function(a):
    sigmoid_value = 1 / (1 + np.exp(-a))
    return sigmoid_value

def calculate_loss(ground_truth, predictions):

    number_of_rows = ground_truth.shape[0]
    loss_value = -(1.0/number_of_rows) * ( np.sum( np.multiply(ground_truth,np.log(predictions)) ) )
    return loss_value

def build_neural_network(X_train, Y_train):
    number_of_rows = X_train.shape[0]
    dimensions = 4
    learning_rate = 1
    number_of_digits = 10
    loss_value_list = []
    loss_value = 0
    m = X_train.shape[1]
    
    w1 = np.random.randn(dimensions,number_of_rows)
    b1 = np.zeros((dimensions,1))
    w2 = np.random.randn(number_of_digits,dimensions)
    b2 = np.random.randn(number_of_digits,1)
    
    for i in range(20):
        activation_value = np.matmul(w1,X_train) + b1#h1
        sigmoid_activation_value = sigmoid_function(activation_value)#z1
        
        activation_value_2 = np.matmul(w2,sigmoid_activation_value) + b2#z2
        
        sigmoid_activation_value_2 = np.exp(activation_value_2) / np.sum(np.exp(activation_value_2), axis=0)#A2

        loss_value = calculate_loss(Y_train, sigmoid_activation_value_2)
        
        d_activation_value_2 = sigmoid_activation_value_2-Y_train
        dw2 = (1.0/m) * np.matmul(d_activation_value_2, sigmoid_activation_value.T)
        db2 = (1.0/m) * np.sum(d_activation_value_2, axis=1, keepdims=True)

        dA1 = np.matmul(w2.T, d_activation_value_2)
        d_activation_value = dA1 * sigmoid_function(activation_value) * (1 - sigmoid_function(activation_value))
        dw1 = (1.0/m) * np.matmul(d_activation_value, X_train.T)
        db1 = (1.0/m) * np.sum(d_activation_value, axis=1, keepdims=True)

        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1

        #if (i % 10 == 0):
        print("Epoch", i, "cost: ", loss_value)
        loss_value_list.append(loss_value)
        
    return w1,w2,b1,b2,loss_value_list

def predict(X_test,Y_test,w1,w2,b1,b2):
    
    Z1 = np.matmul(w1, X_test.T) + b1
    A1 = sigmoid_function(Z1)
    Z2 = np.matmul(w2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    predictions = np.argmax(A2, axis=0)
    losses = calculate_loss(Y_test,predictions)
    
    accuracy = accuracy_score(predictions, Y_test)
    return accuracy,losses
    
def plot_graph(training_loss,validation_loss,test_loss):
    epoch_count =20
    plt.plot(range(epoch_count), training_loss, 'r--')
    plt.plot(range(epoch_count), validation_loss, 'b-')
    plt.plot(range(epoch_count), test_loss, 'g-')
    plt.legend(['Training Loss','Validation Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    
    



# In[125]:



f = gzip.open("mnist.pkl.gz", "rb")
train_set, valid_set, test_set = cPickle.load(f,encoding="latin1")
f.close()

X_train,Y_train = separate_componenets_frame(train_set)


# In[ ]:


w1,w2,b1,b2,loss_value_list = build_neural_network(X_train.T,Y_train.T)

X_test,Y_test = separate_componenets_frame(test_set)
accuracy,losses = predict(X_test,Y_test,w1,w2,b1,b2)
print("Accuracy with Test Set",accuracy)

X_valid,Y_valid = separate_componenets_frame(valid_set)
accuracy_valid,losses_valid = predict(X_valid,Y_valid,w1,w2,b1,b2)
print("Accuracy with Valid Set",accuracy_valid)


# In[ ]:



plt.plot(range(20), loss_value_list, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


# In[ ]:




