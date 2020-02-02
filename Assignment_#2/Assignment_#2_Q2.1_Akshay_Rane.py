#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
# as stated in announcement
# Just do not use the ML models directly. Dated Feb 26, 2019 at 2:58pm
# Using KFold for cross validation
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# In[49]:


column_list = ['id','category']
for i in range(0,30):
    attr_label = 'attr' + str(i+1)
    column_list.append(attr_label)

data = pd.read_csv("wdbc.data" ,  sep=",", names=column_list)

classes = data['category']
y_data = pd.Series(np.where(classes =='M', 1, 0))

X_data = data.drop(['id','category'],axis=1)


def initializeW(dimensions, gradient_type):
    if "SDG" == gradient_type:
        w = [0]*(dimensions) 
    else:
        w = np.zeros(shape=(dimensions, 1))
    return w

def signmoid_funtion(a):
    sigmoid_value = 0
    sigmoid_value = 1 / (1 + np.exp(-a))
    
    return sigmoid_value

    
# Make a prediction with coefficients
def predict_class(row, coefficients):
    prediction = coefficients[0]
    
    for i in range(1, len(row)-1):
        prediction += coefficients[i + 1] * row[i]

    return signmoid_funtion(prediction)


def predict_class_MBG(X_test, coefficients):
    return signmoid_funtion(np.dot(X_test, coefficients))

def evaluate_gradient_SDG(learning_rate, actual_value, prediction):
    gradient_value = learning_rate * (actual_value - prediction) * prediction * (1.0 - prediction)
    return gradient_value 

def evaluate_gradient_MBG(learning_rate, input_batch, prediction,weights):
    
    gradient_value = np.dot(input_batch.T,                             (signmoid_funtion(np.dot(input_batch, weights))                              - prediction)) /prediction.shape[0]
    return (learning_rate * gradient_value) 

# Estimate logistic regression coefficients using stochastic gradient descent
def find_coefficients_SDG(train_X, predictions, learning_rate, n_epoch, w_vector):
    
    
    for epoch in range(n_epoch):
        for index, row in train_X.iterrows():
            prediction_of_class = 0
            row_values = row.values
            real_value = predictions.iloc[index].values[0] 
        
            prediction_of_class = predict_class(row_values, w_vector)
    
            gradient = evaluate_gradient_SDG(learning_rate = learning_rate,                                         actual_value = real_value, prediction = prediction_of_class)
            w_vector[0] = w_vector[0] + gradient
            for i in range(len(row_values)-1):
                w_vector[i+1] = w_vector[i+1] + gradient * row_values[i]
    
    return w_vector

def create_mini_batches(X_data,y_data,batch_size = 32 ): 
    mini_batches_list = [] 
    
    merged_data = np.hstack((X_data, y_data)) 
    np.random.shuffle(merged_data) 
    
    limit_of_batch = merged_data.shape[0] 
    
    i = 0 
    for i in range(limit_of_batch - (batch_size + 1)):
        mini_batch = merged_data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini_batch = mini_batch[:, :-1] 
        Y_mini_batch = mini_batch[:, -1].reshape((-1, 1)) 
        mini_batches_list.append((X_mini_batch, Y_mini_batch))
        
    return mini_batches_list 
    
    

def find_coefficients_MBG(train_X, predictions, learning_rate, n_epoch, w_vector):
    
    for epoch in range(n_epoch):
        batches = create_mini_batches(X_data=train_X,y_data=predictions)
        
        for batch in batches: 
            X_batch, y_batch = batch

            if X_batch.shape[0] == 0:
                continue
            else:
                w_vector = w_vector - evaluate_gradient_MBG(learning_rate = learning_rate,                                         input_batch = X_batch, prediction = y_batch , weights = w_vector) 
    return w_vector
            
    

def normalize_input_data(X_data):
    X_data = (X_data-X_data.min())/(X_data.max()-X_data.min())
    return X_data


def perform_cross_validation(gradient_type , kfold_factor, X_data, predictions):
    
    k_fold = KFold(n_splits=kfold_factor)
    X = np.array(X_data)
    X_shuffled , predictions_shuffled = shuffle(X, predictions, random_state=0)
    accuracy_list = []
    recall_list = []
    precision_list = []
   
    for train_index, test_index in k_fold.split(X_shuffled):
        
        w = initializeW(X_data.shape[1] ,gradient_type)
    
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        
        y_train, y_test = predictions_shuffled[train_index], predictions_shuffled[test_index]
        
        X_train = pd.DataFrame(data=X_train)
        y_train = pd.DataFrame(data=y_train)
        
            
        test_predictions=[]
        actual_predicitions=[]
        
        if "SGD" == gradient_type:
            coefficients = find_coefficients_SDG(train_X = X_train, predictions = y_train, learning_rate = 0.1,                               n_epoch = 100 , w_vector = w)
            
            for row in X_test:
                prediction_class = predict_class(row, coefficients)
                
                test_predictions.append(prediction_class)
        else:
            coefficients = find_coefficients_MBG(train_X = X_train, predictions = y_train, learning_rate = 0.1,                               n_epoch = 100 , w_vector = w)
            y_pred = predict_class_MBG(X_test, coefficients)
            test_predictions =y_pred
                
        counter = 0
        truePositive = 0
        trueNegative = 0
        falseNegative = 0
        falsePositive = 0
        
        for value in y_test:
            pred_value = int(test_predictions[counter])
            
            if pred_value == 1 and value == 1:
                truePositive += 1
            elif pred_value == 0 and value == 0:
                trueNegative += 1
            elif pred_value == 0 and value == 1:
                falseNegative += 1
            elif pred_value == 1 and value == 0 :
                falsePositive += 1
                
            counter = counter + 1 
        
        recall = truePositive / ( truePositive + falseNegative if truePositive + falseNegative else 1) * 100
        accuracy = (truePositive + trueNegative) / ((truePositive + falsePositive + trueNegative +falseNegative)                                                    if (truePositive + falsePositive + trueNegative +falseNegative) else 1) * 100
        precision = truePositive / (truePositive + falsePositive if truePositive + falsePositive else 1) * 100
        
        accuracy_list.append(accuracy)
        recall_list.append(recall)
        precision_list.append(precision)
        
    print("Recall: %.2f" % np.average(recall_list)+str('%'))
    print("Precision: %.2f"% np.average(precision_list)+str('%'))
    print("Accuracy: %.2f" %np.average(accuracy_list)+str('%'))
        
X_data_norm = normalize_input_data(X_data)
print("Stochastic Gradient Descent\n")
perform_cross_validation("SGD" , 10, X_data_norm, y_data)

print("\nMini-Batch Gradient Descent\n")
perform_cross_validation("MBG" , 10, X_data_norm, y_data)


# In[ ]:





# In[ ]:




