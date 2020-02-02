#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np 
import pandas as pd
import re
import math

from sklearn.model_selection import KFold
from sklearn.utils import shuffle


# In[15]:


def perform_preprocessing(train_dataframe, test_dataframe):
    merged_dataset = [train_dataframe, test_dataframe]

    deck_ship = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    ports_ship = {"S": 0, "C": 1, "Q": 2}
    genders = {"male": 0, "female": 1}
    
    mean_age = train_dataframe["Age"].mean()
    std_age = test_dataframe["Age"].std()
    
    
    for dataset in merged_dataset:
        
        dataset["p_relatives"] = dataset["SibSp"] + dataset["Parch"]
        dataset.loc[dataset["p_relatives"] > 0, "individual"] = 0
        dataset.loc[dataset["p_relatives"] == 0, "individual"] = 1
        dataset["individual"] = dataset["individual"].astype(int)
        
        
        dataset["Cabin"] = dataset["Cabin"].fillna("U0")
        dataset["Deck_Pos"] = dataset["Cabin"].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset["Deck_Pos"] = dataset["Deck_Pos"].map(deck_ship)
        dataset["Deck_Pos"] = dataset["Deck_Pos"].fillna(0)
        dataset["Deck_Pos"] = dataset["Deck_Pos"].astype(int)
        
        
        is_null = dataset["Age"].isnull().sum()
        rand_age_to_fill = np.random.randint(mean_age - std_age, mean_age + std_age, size = is_null)
        age_column_copy = dataset["Age"].copy()
        age_column_copy[np.isnan(age_column_copy)] = rand_age_to_fill
        dataset["Age"] = age_column_copy
        dataset["Age"] = train_dataframe["Age"].astype(int)
        
        
        dataset["Embarked"] = dataset["Embarked"].fillna("S")
        dataset["Embarked"] = dataset["Embarked"].map(ports_ship)
        
        dataset["Sex"] = dataset["Sex"].map(genders)
        
        dataset["Fare"] = dataset["Fare"].fillna(0)
        dataset["Fare"] = dataset["Fare"].astype(int)
        
    
    train_dataframe["Age"].isnull().sum()
    merged_dataset = [train_dataframe, test_dataframe]
    
    for dataset in merged_dataset:
        #divide Age into Age Groups
        dataset.loc[ dataset['Age'] <= 10, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5
        dataset.loc[ dataset['Age'] > 60, 'Age'] = 6
        
        #divide Fare into Fare Groups
        dataset.loc[ dataset['Fare'] <= 8, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 15), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 40), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 100), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    
    train_dataframe = train_dataframe.drop(["SibSp"], axis=1)
    test_dataframe = test_dataframe.drop(["SibSp"], axis=1)
    
    
    train_dataframe = train_dataframe.drop(["Parch"], axis=1)
    test_dataframe = test_dataframe.drop(["Parch"], axis=1)
    
    train_dataframe = train_dataframe.drop(["p_relatives"], axis=1)
    test_dataframe = test_dataframe.drop(["p_relatives"], axis=1)
    
    train_dataframe = train_dataframe.drop(["Cabin"], axis=1)
    test_dataframe = test_dataframe.drop(["Cabin"], axis=1)
    
    train_dataframe = train_dataframe.drop(["Name"], axis=1)
    test_dataframe = test_dataframe.drop(["Name"], axis=1)
    
    train_dataframe = train_dataframe.drop(["Ticket"], axis=1)
    test_dataframe = test_dataframe.drop(["Ticket"], axis=1)

    return train_dataframe,test_dataframe


# In[16]:




def perform_decision_tree_algorithm(train_dataframe, test_dataframe, max_depth, min_size):
    
    decision_tree = build_decision_tree(train_dataframe, max_depth, min_size)
    predictions = []
    for index,row in test_dataframe.iterrows():
        prediction = predict(decision_tree, row)
        predictions.append(prediction)
    return(predictions)

def build_decision_tree(train_dataframe, max_depth, min_size):
    root_node , gini_index_columns, information_gain_columns  = perform_split(train_dataframe,performGain = 1)
    display_ig_gini_table(gini_index_columns,information_gain_columns)
    
    split_root_node_recursively(root_node, max_depth, min_size, 1)
    return root_node

def perform_split(train_dataframe, performGain = 0):
    
    gini_index_columns = {}
    information_gain_columns = {}
    number_of_rows = len(train_dataframe)
    
    X_train = train_dataframe.drop("Survived", axis=1)
    Y_train = train_dataframe["Survived"]
    
    class_labels = list(set(label["Survived"] for index,label in train_dataframe.iterrows()))
    
    total_entropy = 0
    column_entropy = 0

    b_column_name, b_column_value, b_score, b_groups = 999, 999, 999, None
    
    number_of_columns = len(X_train.columns)

    for column in X_train.columns:
        
        unique_value = []
        for index, row in train_dataframe.iterrows():
            
            if row[column] in unique_value:
                continue
                
            unique_value.append(row[column])    
            groups = split(column, row[column], train_dataframe)
            
            gini_index = calculate_gini_index(groups, class_labels)
            
            if performGain == 1:
                total_entropy = calculate_total_entropy(train_dataframe, class_labels, number_of_rows)
                column_entropy = calculate_column_entropy(train_dataframe, column, class_labels, number_of_rows)
            
                information_gain_columns[column] = total_entropy - column_entropy
                gini_index_columns[column] = gini_index
            
            if gini_index < b_score:
                b_column_name, b_column_value, b_score, b_groups = column, row[column], gini_index, groups
    
    if performGain == 1:
        return {'index':b_column_name, 'value':b_column_value, 'groups':b_groups}, gini_index_columns, information_gain_columns
    else:
        return {'index':b_column_name, 'value':b_column_value, 'groups':b_groups}
        
    
# Split a dataset based on an attribute and an attribute value
def split(column_index, value, train_dataframe):
    
    left_split = pd.DataFrame(columns=train_dataframe.columns)
    right_split = pd.DataFrame(columns=train_dataframe.columns)
    
    for index, row in train_dataframe.iterrows():
        if row[column_index] < value:
            left_split = left_split.append(row)
        else:
            right_split = right_split.append(row)
    
    return left_split, right_split

def calculate_gini_index(groups, class_labels):
    total_instance_groups = 0.0
    
    for group in groups:
        total_instance_groups = total_instance_groups +  len(group)
    
    gini_index = 0.0
    
    for group in groups:
        group_size = float(len(group))
        if group_size == 0:
            continue
            
        score = 0.0
        
        for label in class_labels:
            proportion = [row["Survived"] for index,row in group.iterrows()].count(label) / group_size
            score = score + proportion * proportion
        
        gini_index = gini_index + (1.0 - score) * (group_size / total_instance_groups)
        
    return gini_index

def calculate_total_entropy(train_dataframe, class_labels , number_of_rows):
    total_entropy = 0.0
    
    for label in class_labels:
        total_entropy = total_entropy + (-(len(train_dataframe[train_dataframe['Survived'] == label])                                           /number_of_rows)*math.log2(len(train_dataframe[train_dataframe['Survived']                                                                                           == label])/number_of_rows))
    return total_entropy

def calculate_column_entropy(train_dataframe,column_name,class_labels,number_of_rows):
    column_entropy = 0
    
    unique_column_values = list(train_dataframe[column_name].unique())
    
    for column_value in unique_column_values:
        for label in class_labels:
            
            probability_value = -len(train_dataframe[train_dataframe[column_name] == column_value])/number_of_rows
            conditional_probability_value = len(train_dataframe[(train_dataframe['Survived'] == label) & (train_dataframe[column_name]                                                                       == column_value)])/len(train_dataframe[column_name] == column_value)
            if conditional_probability_value != 0:
                column_entropy = column_entropy + probability_value*(conditional_probability_value*math.log2(conditional_probability_value))
            else:
                continue
                
    return column_entropy  

def display_ig_gini_table(gini_index_columns,information_gain_columns):
    print("%12s\t%12s\t%12s"%("Column_Name","Information_Gain","Gini_Index"))

    for column_name, gain_value in information_gain_columns.items():
        print("%12s\t%12s\t%12s"% (column_name,gain_value,gini_index_columns[column_name]))
    
        

def split_root_node_recursively(current_node, max_depth, min_size, depth):
    left, right = current_node["groups"]
    del(current_node["groups"])
    
    if len(left) == 0 or len(right) == 0:
        if len(left) == 0:
            current_node["left_tree"] = current_node["right_tree"] = create_leaf_node(right)
        else:
            current_node["left_tree"] = current_node["right_tree"] = create_leaf_node(left)
            
        return

    if depth >= max_depth:
        current_node["left_tree"], current_node["right_tree"] = create_leaf_node(left), create_leaf_node(right)
        return

    if len(left) <= min_size:
        current_node["left_tree"] = create_leaf_node(left)
    else:
        current_node["left_tree"] = perform_split(left)
        split_root_node_recursively(current_node["left_tree"], max_depth, min_size, depth+1)

    if len(right) <= min_size:
        current_node["right_tree"] = create_leaf_node(right)
    else:
        current_node["right_tree"] = perform_split(right)
        split_root_node_recursively(current_node["right_tree"], max_depth, min_size, depth+1)

def create_leaf_node(group):
    classifier_count = [row["Survived"] for index,row in group.iterrows()]
    return max(set(classifier_count), key=classifier_count.count)
    
def predict(current_node, row):
    
    if row[current_node["index"]] < current_node["value"]:
        if isinstance(current_node["left_tree"], dict):
            return predict(current_node["left_tree"], row)
        else:
            return current_node["left_tree"]
    else:
        if isinstance(current_node["right_tree"], dict):
            return predict(current_node["right_tree"], row)
        else:
            return current_node["right_tree"]

def calculate_accuracy(ground_truth, predicted):
    correct_prediction = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predicted[i]:
            correct_prediction = correct_prediction + 1
    return correct_prediction / float(len(ground_truth)) * 100.0

def perform_cross_validation(kfold_factor, train_dataframe, test_dataframe):
    k_fold = KFold(n_splits=kfold_factor)
    
    X_train_dataframe = train_dataframe.drop("Survived",axis = 1)
    Y_train_dataframe = train_dataframe[["Survived","PassengerId"]]
    
    
    X_column_values = X_train_dataframe.columns
    
    Y_column_values = Y_train_dataframe.columns
    
    X_train_dataframe = np.array(X_train_dataframe)
    Y_train_dataframe = np.array(Y_train_dataframe)
    
    
    X_shuffled_train , predictions_shuffled_train = shuffle(X_train_dataframe, Y_train_dataframe, random_state=0)
    
    accuracy_array = []
    
    for train_index, test_index in k_fold.split(X_shuffled_train):
        y_train, y_test = predictions_shuffled_train[train_index], predictions_shuffled_train[test_index]
       
        X_train, X_test = X_shuffled_train[train_index], X_shuffled_train[test_index]
       
        
        X_train = pd.DataFrame(data=X_train, columns=X_column_values, dtype =object)
        y_train = pd.DataFrame(data=y_train, columns=Y_column_values, dtype =object)
        
    
        train_dataframe_new = pd.merge(X_train, y_train, on=['PassengerId'])
        train_dataframe_new = train_dataframe_new.drop(["PassengerId"], axis=1)
        
        X_test = pd.DataFrame(data=X_test, columns=X_column_values, dtype =object)
        y_test = pd.DataFrame(data=y_test, columns=Y_column_values, dtype =object)
        
        test_dataframe_new = pd.merge(X_test, y_test, on=['PassengerId'])
        test_dataframe_new = test_dataframe_new.drop(["PassengerId"], axis=1)
        
        
        predictions = perform_decision_tree_algorithm(train_dataframe_new, test_dataframe_new, 5, 10)
        accuracy = calculate_accuracy(y_test.Survived,predictions)
        accuracy_array.append(accuracy)
        
    return max(accuracy_array)
        
   


# In[17]:



train_dataframe = pd.read_csv("train.csv")
test_dataframe = pd.read_csv("test.csv")
Y_test = pd.read_csv("gender_submission.csv")
test_dataframe = pd.merge(test_dataframe, Y_test, on=['PassengerId'])
train_dataframe, test_dataframe = perform_preprocessing(train_dataframe,test_dataframe)

# cross validation remaining

accuracy = perform_cross_validation(5, train_dataframe, test_dataframe)
print("Accuracy: %.3f%%" % accuracy)


# In[ ]:




