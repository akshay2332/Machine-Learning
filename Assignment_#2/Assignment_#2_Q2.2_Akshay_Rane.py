#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
# as stated in announcement
# Just do not use the ML models directly. Dated Feb 26, 2019 at 2:58pm
# Using KFold for cross validation
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# In[78]:


column_list = ['id','category']
for i in range(0,30):
    attr_label = 'attr' + str(i+1)
    column_list.append(attr_label)

data_pgm = pd.read_csv("wdbc.data" ,  sep=",", names=column_list)

y_data_pgm = data_pgm['category']

X_data_pgm = data_pgm.drop(['id','category'],axis=1)


# In[82]:


def signmoid_funtion(a):
    sigmoid_value = 0
    sigmoid_value = 1 / (1 + np.exp(-a))    
    return sigmoid_value

def prediction(weight, imput_data, w0,probablity_of_M, probablity_of_B,                mean_vectors_classes, count_vectors_classes):
    first_term = np.dot(weight.T,imput_data)
    argument_a = first_term + w0
    return signmoid_funtion(argument_a.astype(float))

def determine_covariance(input_data,mean_vectors_classes,count_vectors_classes):
    
    count_m = count_vectors_classes["M"]
    count_b = count_vectors_classes["B"]
    total_count = count_m + count_b
    
    column_length = input_data.shape[1]
    
    for index, row in input_data.iterrows():
        s1 = s2 = np.zeros((column_length-1,1), dtype=object)
        
        row_matrix = row.values[0:(column_length-1)].reshape(column_length-1,1)
        category_mean_vector = mean_vectors_classes[row.category]
        
        if "M" == row.category:
            category_mean_vector =  np.reshape(category_mean_vector, (column_length-1, 1))
            diff_matrix = row_matrix - category_mean_vector
            s1 = s1 + np.dot(diff_matrix,diff_matrix.T)
        else:
            category_mean_vector =  np.reshape(category_mean_vector, (column_length-1, 1))
            diff_matrix = row_matrix - category_mean_vector
            s2 = s2 + np.dot(diff_matrix,diff_matrix.T)
    
    return (((s1)/total_count))+(((s2)/total_count))

def determine_coefficients(probablity_of_M,probablity_of_B,merge_data_frame,                            mean_vectors_classes,count_vectors_classes):
    
    covariance_matrix = determine_covariance(merge_data_frame,mean_vectors_classes,count_vectors_classes)
   
    mean_vector_class_m = mean_vectors_classes["M"]
    mean_vector_class_b = mean_vectors_classes["B"]
    
    mean_vector_class_m = np.reshape(mean_vector_class_m, (merge_data_frame.shape[1] -1, 1))
    mean_vector_class_b= np.reshape(mean_vector_class_b, (merge_data_frame.shape[1] -1 , 1))


    covariance_matrix_inv = np.linalg.inv(covariance_matrix.astype("float"))
    
    w =  covariance_matrix_inv.dot(( mean_vector_class_m- mean_vector_class_b))
    
    w0 = ((-1) * mean_vector_class_m.T.dot(covariance_matrix).dot(mean_vector_class_m)) +     ((1) * mean_vector_class_b.T.dot(covariance_matrix).dot(mean_vector_class_b)) +     np.log(probablity_of_M/probablity_of_B)
   
    return w,w0

def perform_cross_validation(kfold_factor, X_data_pgm, predictions_pgm):
    
    # 1 is M
    # 2 is B
    k_fold = KFold(n_splits=kfold_factor)
    X_pgm = np.array(X_data_pgm)
    X_shuffled_pgm , predictions_shuffled_pgm = shuffle(X_pgm, predictions_pgm, random_state=0)
    
    accuracy_list = []
    recall_list = []
    precision_list = []
    
    for train_index, test_index in k_fold.split(X_shuffled_pgm):
        X_train_pgm, X_test_pgm = X_shuffled_pgm[train_index], X_shuffled_pgm[test_index]
        y_train_pgm, y_test_pgm = predictions_shuffled_pgm[train_index], predictions_shuffled_pgm[test_index]
        
        X_merge_train = pd.DataFrame(data=X_train_pgm, columns=column_list[2:], dtype =object)
        y_merge_train = pd.DataFrame(data=y_train_pgm, columns= ["category"], dtype =object)
        
        X_test_pgm = pd.DataFrame(data=X_test_pgm, columns=column_list[2:] , dtype =object)
        
        merge_data_frame = X_merge_train
        merge_data_frame["category"] = y_merge_train
        merge_data_frame = merge_data_frame.dropna()
        
        grouped = merge_data_frame.groupby('category')
        
        mean_vectors_classes = {}
        count_vectors_classes = {}

        for name, group in grouped:
            group_mean_value = group.mean()
            count_vectors_classes[name] = group.count()["attr1"]
            mean_vectors_classes[name] = [v for (k,v) in group_mean_value.items()]
        
        probablity_of_M = count_vectors_classes["M"] / (count_vectors_classes["M"] + count_vectors_classes["B"])        
        probablity_of_B = count_vectors_classes["B"] / (count_vectors_classes["M"] + count_vectors_classes["B"])
        
        w,w0 = determine_coefficients(probablity_of_M,probablity_of_B,merge_data_frame,                                      mean_vectors_classes,count_vectors_classes)
        
        predicted_class = []
        
        for index,row in X_test_pgm.iterrows():
            
            probabiity_of_class_m = prediction(weight= w, imput_data = row.values, w0 = w0,                                              probablity_of_M = probablity_of_M, probablity_of_B = probablity_of_B,                                      mean_vectors_classes = mean_vectors_classes, count_vectors_classes = count_vectors_classes)
            if probabiity_of_class_m > 0.5:
                predicted_class.append("M")
            else:
                predicted_class.append("B") 
                
        counter = 0
        truePositive = 0
        trueNegative = 0
        falseNegative = 0
        falsePositive = 0
        
        for value in y_test_pgm:
            pred_value = predicted_class[counter]
            
            if pred_value == "M" and value == "M":
                truePositive += 1
            elif pred_value == "B" and value == "B":
                trueNegative += 1
            elif pred_value == "B" and value == "M":
                falseNegative += 1
            elif pred_value == "M" and value == "B" :
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
                
        
perform_cross_validation(kfold_factor = 5, X_data_pgm = X_data_pgm, predictions_pgm = y_data_pgm)


# In[ ]:





# In[ ]:




