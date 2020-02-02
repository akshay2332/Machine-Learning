#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# In[2]:


def load_yeast_dataset(file_name):
    yeast_dataframe = pd.read_table(file_name, delim_whitespace=True, names=("id", "feature_1", "feature_2",                                                                     "feature_3","feature_4","feature_5",                                                                     "feature_6","feature_7","feature_8","cluster_name"))
    return yeast_dataframe


def find_euclidean_distance(sample_1, sample_2, dimensions):
    distance = 0
 
    for feature in dimensions:
        distance += pow((sample_1[feature] - sample_2[feature]), 2)
    return math.sqrt(distance)

def find_nearest_neighbors(neighbors_of_sample, sample, number_of_neighbors,dimensions):
    
    distance_of_neighbor = []
    
    for idx,datapoint in neighbors_of_sample.iterrows():
        dist = find_euclidean_distance(sample, datapoint, dimensions)
        distance_of_neighbor.append((datapoint, dist))
    
    distance_of_neighbor.sort(key=operator.itemgetter(1))
    neighbors = []
    
    for x in range(number_of_neighbors):
        neighbors.append(distance_of_neighbor[x][0])
    
    return neighbors

def predict_class(neighbors):
    
    classVotes = {}
    for neighbor in neighbors:
        
        response = neighbor.cluster_name
        
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
                          

def fetch_accuracy(Y_test, predictions):
    correct_classification = 0
    counter = 0
    
    for idx, row in Y_test.iterrows():
        if row.cluster_name == predictions[counter]:
            correct_classification += 1
        counter = counter + 1
                          
    return (correct_classification/float(len(Y_test))) * 100.0

def perform_cross_validation_kmeans(kfold_factor, yeast_dataset, neighbors):
    
    k_fold = KFold(n_splits=kfold_factor)
    
    Y_dataframe = yeast_dataset[["id","cluster_name"]]
    X_dataframe = yeast_dataset.drop("cluster_name",axis = 1)
    
    X_column_values = X_dataframe.columns
    Y_column_values = Y_dataframe.columns
    
    X_dataframe = np.array(X_dataframe)
    Y_dataframe = np.array(Y_dataframe)
    
    X_shuffled , predictions_shuffled = shuffle(X_dataframe, Y_dataframe, random_state=0)
    
    accuracies = []
    
    for train_index, test_index in k_fold.split(X_shuffled):
        y_train, y_test = predictions_shuffled[train_index], predictions_shuffled[test_index]
       
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        
        X_train = pd.DataFrame(data=X_train, columns=X_column_values, dtype =object)
        Y_train = pd.DataFrame(data=y_train, columns=Y_column_values, dtype =object)
        
        X_test = pd.DataFrame(data=X_test, columns=X_column_values, dtype =object)
        Y_test = pd.DataFrame(data=y_test, columns=Y_column_values, dtype =object)
        
        train_dataframe_new = pd.merge(X_train, Y_train, on=["id"])
        train_dataframe_new = train_dataframe_new.drop(["id"], axis=1)
        X_test = X_test.drop(["id"],axis=1)
        
        predictions=[]               
        
        for idx,sample in X_test.iterrows():
            nearest_neighbors = find_nearest_neighbors(train_dataframe_new, sample, neighbors,X_test.columns)
            #print("nearest_neighbors",nearest_neighbors)
            #print("sample",sample)
            predictied_class = predict_class(nearest_neighbors)
            
            predictions.append(predictied_class)
          
        accuracy = fetch_accuracy(Y_test, predictions)
        accuracies.append(accuracy)
    
    return np.max(accuracies)
        
    


# In[3]:


yeast_dataframe = load_yeast_dataset("yeast.data")

accuracy = perform_cross_validation_kmeans(10, yeast_dataframe,10)

print("Maximum accuracy", accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




