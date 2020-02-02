#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal


# In[2]:


def load_dataframe(k):
    with open('multigauss.txt') as f:
        file = f.read()
        array_data = (file.split("\n"))
        counter = 0
        array_data = array_data[4:]
        dimension_1=[]
    
        for data in array_data:
            temp_array = data.strip().split(" ")
            if len(temp_array) >1:
                dimension_1.append([float(temp_array[0]),float(temp_array[1])])
            
    dataframe = np.array([dimension_1])[0]
        
    n = dataframe.shape[0]
    f = dataframe.shape[1]
    
    w = np.random.rand(n,k)
    
    pi = np.random.rand(k,1)
    #μ is a k × f matrix containing the means of each gaussian
    #mu = np.random.rand(k,f)
    #Σ is an f × f × k tensor of covariance matrices. Σ(:, :, i) is the covariance of gaussian i
   
    mu = np.asmatrix(np.random.random((k,f)))
    covariance = np.array([np.asmatrix(np.identity(f)) for i in range(k)])
    
    return dataframe, k, pi, mu, covariance, w, n, f

        

def expectation(X, k, pi, mu, covariance):
    summation_gaussian = 0.0
    data = X.copy()
    n = X.shape[0]
    
    for sample in range(n):
        summation_gaussian = 0
        for gaussian in range(k):
            value_of_gaussian = multivariate_normal.pdf(data[sample, :],mu[gaussian].A1,                                                        covariance[gaussian]) * pi[gaussian]
            summation_gaussian += value_of_gaussian
            w[sample, gaussian] = value_of_gaussian
        
        w[sample, :] /= summation_gaussian
    
    return w
    
def maximize_mean(X,k,w,f):
    new_mu_gaussian = []
    n = X.shape[0]
    for gaussian in range(k):
        responsibilities = w[: ,gaussian].sum()
        
        pi[gaussian] = 1/n * responsibilities
        new_mu = np.zeros(f)
        new_covariance = np.zeros((f,f))

        for sample in range(n):
            new_mu += (X[sample, :] * w[sample,gaussian])
            
         #   new_covariance += w[sample, gaussian] * ((X[sample, :] - mu[gaussian, :]).T *\
         #                                           (X[sample, :] - mu[gaussian, :]))
              
        new_mu_gaussian.append(new_mu / responsibilities)
        #covariance[gaussian] = new_covariance / responsibilities
    return new_mu_gaussian

def maximize_covariance(X,k,w,mu,f):
    new_covariance_gaussian = []
    
    for gaussian in range(k):
        responsibilities = w[: ,gaussian].sum()
        
        pi[gaussian] = 1/n * responsibilities
        new_mu = np.zeros(f)
        new_covariance = np.zeros((f,f))

        for sample in range(n):
            #new_mu += (X[sample, :] * w[sample,gaussian])
            
            new_covariance += w[sample, gaussian] * ((X[sample, :] - mu[gaussian, :]).T *                                                    (X[sample, :] - mu[gaussian, :]))
        
        new_covariance_gaussian.append(  new_covariance / responsibilities )  
       # mu[gaussian] = new_mu / responsibilities
       # covariance[gaussian] = 
    
    return new_covariance_gaussian
    
    

def maximize_mixtures(k, w, X, pi, mu,n):
    log_likelihood = 0
    for sample in range(n):
        summation_gaussian_value = 0
        for gaussian in range(k):
            summation_gaussian_value += multivariate_normal.pdf(X[sample, :],mu[gaussian, :].A1,                                                                         covariance[gaussian, :]) * pi[gaussian]
        
        log_likelihood += np.log(summation_gaussian_value) 
    
    return log_likelihood


def perform_em(X, k, pi, mu, covariance, nIter, w, n, f):
    
    number_iteration = 0
    log_likelihood = 1
    previous_log_likelihood = 0
    
    while((log_likelihood - previous_log_likelihood) > 1e-4):
        number_iteration = number_iteration + 1
        previous_log_likelihood = maximize_mixtures(k, w, X, pi, mu,n)
        w = expectation(X, k, pi, mu, covariance)
        new_means_gaussian = maximize_mean(X,k,w,f)
        new_covariance = maximize_covariance(X,k,w,mu,f)
        
        for gaussian in range(k):
            mu[gaussian] = new_means_gaussian[gaussian]
            covariance[gaussian] = new_covariance[gaussian]
        
        log_likelihood = maximize_mixtures(k, w, X, pi, mu,n)
        print('Iteration %d: log-likelihood is %.6f'%(number_iteration, log_likelihood))
        
    
        


# In[3]:


k = 5
dataframe, k, pi, mu, covariance, w, n, f = load_dataframe(k)
perform_em(dataframe, k, pi, mu, covariance, 10, w, n, f)



