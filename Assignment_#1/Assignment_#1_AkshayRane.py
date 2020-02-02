#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Assignment 1
# # Facebook dataset

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[8]:


# reading training dataset
#location of dataset at Dataset/Training/Features_Variant_1.csv
train_data = pd.read_csv("Dataset/Training/Features_Variant_1.csv")


# In[9]:



train_data.drop(train_data.iloc[:,5:29], inplace=True, axis=1)
train_data = train_data.loc[:, (train_data != 0).any(axis=0)]


# In[10]:


def perform_cross_validation(model_type , kfold_factor):
    
    regression_errors = []
    k_fold = KFold(n_splits=kfold_factor)
    
    X = np.array(train_data.iloc[:,:-1])
    predictions = train_data.iloc[:,-1]
    
    X_shuffled , predictions_shuffled = shuffle(X, predictions, random_state=0)
    
    if model_type.lower() == "lasso":
        regression_model = Lasso(alpha=1.0, normalize=True)
    elif model_type.lower() == "ridge":
        regression_model =  Ridge(alpha=1.0,normalize=True)
    else:
        return
        
    
    for train_index, test_index in k_fold.split(X_shuffled):
        X_train, X_test = X_shuffled[train_index], X_shuffled[test_index]
        y_train, y_test = predictions_shuffled[train_index], predictions_shuffled[test_index]

        regression_model.fit(X_train, y_train)

        y_regression_prediction = regression_model.predict(X_test)
        
        mse_regression = np.mean((y_regression_prediction - y_test)**2)
        regression_errors.append(mse_regression) 
    
    avegrage_regression_error=np.average(regression_errors)
    print ( ("Average Mean Square Error for model type {} is {} :").format( model_type , avegrage_regression_error ) ) 
    
    
    


# In[11]:


perform_cross_validation("ridge",5)
perform_cross_validation("lasso",5)

perform_cross_validation("ridge",10)
perform_cross_validation("lasso",10)

perform_cross_validation("ridge",15)
perform_cross_validation("lasso",15)


# In[ ]:




