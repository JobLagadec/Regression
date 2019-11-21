import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, tree, neural_network
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


""" auteur: Pierre-Adrien """
def get_train_test_sets(x_data, y_data, train_ratio = 0.75) :
    (nb_rows, nb_col) = np.shape(x_data)
    
    # shuffle data, necessaire si données déjà ordonnées
    y_data = np.reshape(y_data, (len(y_data),1)) # nécessaire pour concaténation 
    data = np.concatenate((x_data,y_data),axis = 1)
    np.random.shuffle(data)
    
    train_set_size = int(train_ratio * nb_rows)
    
    train_set = data[0:train_set_size,:]
    test_set = data[train_set_size:,:]
    x_train_set , y_train_set = train_set[:,:nb_col] , train_set[:,nb_col]
    x_test_set , y_test_set = test_set[:,:nb_col] , test_set[:,nb_col]
    
    return x_train_set , y_train_set , x_test_set , y_test_set

""" auteur: Pierre-Adrien """

""" Pour les regression
reg.coef_ = coefficients directeur
reg.intercept_ = ordonnée à l'origine
reg.score(x_test_set,y_test_set) = coefficient of determination R^2 of the prediction
reg.predict(x_test_set) = prediction of the test set depending on the trained model
"""
def Least_Squares_Regression(x_train_set , y_train_set , x_test_set , y_test_set):
    reg = linear_model.LinearRegression()
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.coef_ , reg.intercept_ , reg.score(x_test_set,y_test_set), reg.predict(x_test_set)

def Ridge_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set):
    reg = linear_model.RidgeCV()
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.coef_ , reg.intercept_ , reg.score(x_test_set,y_test_set), reg.predict(x_test_set)

def Lasso_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set):
    reg = linear_model.LassoCV(cv=5)
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.coef_ , reg.intercept_ , reg.score(x_test_set,y_test_set), reg.predict(x_test_set)

def SVM_Regression(x_train_set , y_train_set , x_test_set , y_test_set,kernel="linear"):
    reg = svm.SVR(gamma='auto',kernel=kernel)
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.score(x_test_set,y_test_set), reg.predict(x_test_set)

def Tree_Regression(x_train_set , y_train_set , x_test_set , y_test_set):
    reg = tree.DecisionTreeRegressor()
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.score(x_test_set,y_test_set), reg.predict(x_test_set)

def NN_Regression(x_train_set , y_train_set , x_test_set , y_test_set):
    reg = neural_network.MLPRegressor()
    reg.fit(x_train_set, y_train_set) # train the model
    return reg.score(x_test_set,y_test_set), reg.predict(x_test_set)


(x,y) = get_trainable_data(housing_file)
x_train_set , y_train_set , x_test_set , y_test_set = get_train_test_sets(x, y, train_ratio = 0.75)

""" auteur: Pierre-Adrien """
def Matrix_Plot(x_data,y_data): ## !! seulement pour housing.data
    col_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
    y_data = np.reshape(y_data, (len(y_data),1)) # nécessaire pour concaténation 
    data = np.concatenate((x_data,y_data),axis = 1)
    xy_df = pda.DataFrame(data, columns=col_name)
    pda.plotting.scatter_matrix(xy_df, alpha=1, figsize=(15, 15))
    plt.show()
    
def Correlation_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # nécessaire pour concaténation 
    data = np.concatenate((x_data,y_data),axis = 1)
    return np.corrcoef(np.transpose(data))

def Covariance_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # nécessaire pour concaténation 
    data = np.concatenate((x_data,y_data),axis = 1)
    return np.cov(np.transpose(data))
    





