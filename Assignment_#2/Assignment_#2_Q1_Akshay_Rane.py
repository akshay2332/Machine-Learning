#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np


# In[32]:


class LinearDiscriminant(object):
    
    def fit(self,data):
        grouped=data.groupby('category')

        X_vector = data[["sepal_length","sepal_width","petal_height","petal_width"]].values
        class_categories = data['category'].values
        
        
        feature_count = X_vector.shape[1]
        
        mean_vectors_classes = {}
        count_vectors_classes = {}

        for name, group in grouped:
            group_mean_value = group.mean()
            count_vectors_classes[name] = group.count()["sepal_length"]
            mean_vectors_classes[name] = [group_mean_value["sepal_length"],group_mean_value["sepal_width"] ,                                 group_mean_value["petal_height"],group_mean_value["petal_width"]]

        # Calculating Within-class scatter matrix S_W 
        withinClassScatter = np.zeros((feature_count,feature_count))

        
        4
        counter = 0
        withinClassScatterValue = {}

        for row in X_vector:
            row_catergory = class_categories[counter]
            category_mean_vector = mean_vectors_classes[row_catergory]
    
            row = row.reshape(feature_count,1)
            category_mean_vector =  np.reshape(category_mean_vector, (feature_count, 1))
            current_class_scatter_matrix = np.zeros((feature_count,feature_count))
    
            if row_catergory in withinClassScatterValue:
                current_class_scatter_matrix += withinClassScatterValue[row_catergory]
                mean_row_diff = row - category_mean_vector
                current_class_scatter_matrix = current_class_scatter_matrix + (mean_row_diff).dot((mean_row_diff).T)
            else:
                mean_row_diff = row - category_mean_vector
                current_class_scatter_matrix += (mean_row_diff).dot((mean_row_diff).T)
        
            withinClassScatterValue[row_catergory] = current_class_scatter_matrix
            counter = counter+1
    
        for class_name, scatter_of_each_category in withinClassScatterValue.items():
            withinClassScatter = withinClassScatter + scatter_of_each_category
        
        print ("Within-class scatter matrix S_W \n",withinClassScatter)


        # Calculating between-class scatter matrix S_B
        betweenClassScatter = np.zeros((4,4))
        mean_of_all_samples = np.mean(X_vector, axis=0)

        for class_name, mean_vector in mean_vectors_classes.items():
            samples_for_class = count_vectors_classes[class_name]
            mean_vector = np.reshape(mean_vector, (feature_count, 1))
            mean_of_all_samples = mean_of_all_samples.reshape(feature_count,1)
            mean_vec_diff = mean_vector - mean_of_all_samples
            betweenClassScatter += samples_for_class * (mean_vec_diff).dot((mean_vec_diff).T)

        print('between-class Scatter Matrix:\n', betweenClassScatter)

        # Solving for eigenvalue for the matrix S^−1_WS_B  to obtain the linear discriminants.
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(withinClassScatter).dot(betweenClassScatter))
        print("\n")
        counter = 0

        for values in eigen_values:
            eigvec_sc = eigen_vectors[:,counter].reshape(feature_count,1)   
            print("Eigenvector {}: \n{}".format(counter+1, eigvec_sc))
            print("Eigenvalue {:}: {:.2e}\n".format(counter+1, values))
            counter = counter+1
            


# In[33]:


linearDiscriminant = LinearDiscriminant()

# reading the dataset
# sepal length in cm 
# sepal width in cm 
# petal length in cm 
# petal width in cm 
data = pd.read_csv("iris.data" ,  sep="," , names=["sepal_length","sepal_width","petal_height",                                                        "petal_width","category"] )
linearDiscriminant.fit(data)


# In[ ]:





# In[ ]:




