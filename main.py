import numpy as np
import matplotlib.pyplot as plt
import sklearn
from math import *
import pandas as pda
from sklearn.preprocessing import scale

housing_file = "data_regression/HousingData.csv"
prostate_file = "data_regression/prostate.data"

"""auteur: Alexis """

def load_csv(file_name):
    csv_data = pda.read_csv(file_name)
    data = csv_data.values
    targets = data[:,-1]
    data = data[:,:-1]
    return data, targets

#housing_data = load_csv(housing_file)

"""auteur: Alexis """
def mean(column):
    l = len(column)
    mean = 0
    for k in range(l):
        if(np.isnan(column[k]) == False):
            mean += column[k]
    return mean/l

"""auteur: Alexis """
def clean_csv_data(data):
    n, m = data.shape
    res = data
    means = np.zeros(m)    
    for k in range(m):
        means[k] = mean(data[:,k])
    #For each individuals and for each feature, if the value of the feature is unknown, we replace it by the mean
    for i in range(n):
        for j in range(m):
            if (np.isnan(data[i][j])):
                res[i][j] = means[j]
    return res

"""auteur: Alexis """   
def normalize_data(data):
    res = data
    n, m = data.shape
    maxes = np.zeros(m)
    for k in range(m):
        maxes[k] = max(data[:,k])
        res[:,k] = res[:,k]/maxes[k]
    return res

#cleaned_housing_data = clean_csv_data(housing_data)
#normalized_housing_data = normalize_data(cleaned_housing_data)


"""
auteur : Tom Dauvé
"""
def load_data_data(filepath):
    df = pda.read_table(filepath)
    data = df.values
    output  = []
    for i in range(len(data)):
        output.append(data[i][1:])
    return output

"""
auteur : Tom Dauvé
"""
def clean_data_data(data):
    for i in range(len(data)):
        if(data[i][-1] == 'T'):
            data[i][-1] = 1
        else:
            data[i][-1] = 0
    return np.array(data)


prostate_data = load_data_data(prostate_file)
cleaned_prostate_data =  clean_data_data(prostate_data)
normalized_prostate_data = normalize_data(cleaned_prostate_data)

"""auteur: Alexis """ 
def get_trainable_data(file):
    if (".data" in file):
        prostate_data = load_data_data(prostate_file)
        cleaned_prostate_data =  clean_data_data(prostate_data)
        normalized_prostate_data = normalize_data(cleaned_prostate_data)
        x = normalized_prostate_data
    elif(".csv" in file):
        housing_data, y = load_csv(housing_file)
        cleaned_housing_data = clean_csv_data(housing_data)
        x = normalize_data(cleaned_housing_data)
    else:
        return ("file format not supported yet by this code, our engineers are currently working on it")
    
    return (x,y)

