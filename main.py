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
auteur : Tom Dauve
"""
def load_data_data(filepath):
    df = pda.read_table(filepath)
    data = df.values
    output  = data[:,:9]
    #for i in range(len(data)):
        #output.append(data[i][1:len(data[i])-1])
    y = data[:,9]
    return np.array(output), np.array(y)

"""
auteur : Tom Dauve
"""
#this function should not be used.
def clean_data_data(data):
    for i in range(len(data)):
        if(data[i][-1] == 'T'):
            data[i][-1] = 1
        else:
            data[i][-1] = 0
    return np.array(data, dtype = float)


prostate_data, y = load_data_data(prostate_file)
normalized_prostate_data = normalize_data(prostate_data)

"""auteur: Alexis """ 
def get_trainable_data(file):
    if (".data" in file):
        cleaned_prostate_data, y = load_data_data(file)
        normalized_prostate_data = normalize_data(cleaned_prostate_data)
        x = normalized_prostate_data
    elif(".csv" in file):
        housing_data, y = load_csv(file)
        cleaned_housing_data = clean_csv_data(housing_data)
        x = normalize_data(cleaned_housing_data)
    else:
        return ("file format not supported yet by this code, our engineers are currently working on it")
    
    return (x,y)


""" auteur: Pierre-Adrien """
def get_train_test_sets(x_data, y_data, train_ratio = 0.75) :
    (nb_rows, nb_col) = np.shape(x_data)
    
    # shuffle data, necessaire si donnees deja ordonnees
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation 
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
reg.intercept_ = ordonnee a l'origine
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
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation 
    data = np.concatenate((x_data,y_data),axis = 1)
    xy_df = pda.DataFrame(data, columns=col_name)
    pda.plotting.scatter_matrix(xy_df, alpha=1, figsize=(15, 15))
    plt.show()
    
def Correlation_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation 
    data = np.concatenate((x_data,y_data),axis = 1)
    return np.corrcoef(np.transpose(data))

def Covariance_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation  
    data = np.concatenate((x_data,y_data),axis = 1)
    return np.cov(np.transpose(data))

"""auteur: Alexis """       
#Function to approximate a gaussian (evaluated on x) that fits the points in data
def kde(x, data, h=0):
    estimated_mean = 0
    estimated_variance = 0
    n = len(data)
    for k in range(n):
        estimated_mean += data[k]
    estimated_mean = estimated_mean/n
    for i in range(n):
        estimated_variance += (data[i] - estimated_mean/n)**2
    estimated_variance = estimated_variance/n
    if (h == 0):
        h = 1.06*sqrt(estimated_variance)*n**(-1/5)
    L = len(x)
    y = np.zeros(L)
    for j in range(L):
        su = 0
        for k in range(n):
            su += exp(-(((x[j] - data[k])/h - estimated_mean)**2)/(2*estimated_variance))/(sqrt(2*np.pi*estimated_variance))
        y[j] = su/(n*h)
    return y

"""auteur: Alexis """   
#gaussian kernel density estimator evaluated on x, that fits the relationship between data1 and data2 
def kde_2D(x, data1, data2, h=0):
    estimated_mean = 0
    estimated_variance = 0
    n = len(data1)
    for k in range(n):
        estimated_mean += data1[k]
    estimated_mean = estimated_mean/n
    for i in range(n):
        estimated_variance += (data1[i] - estimated_mean/n)**2
    estimated_variance = estimated_variance/n
    if (h == 0):
        h = 1.06*sqrt(estimated_variance)*n**(-1/5)
    L = len(x)
    y = np.zeros(L)
    for j in range(L):
        sum1 = 0
        sum2 = 0
        for k in range(n):
            sum1 += data2[k]*np.exp(-(((x[j] - data1[k])/h - estimated_mean)**2)/(2*estimated_variance))/(sqrt(2*np.pi*estimated_variance))
            sum2 += np.exp(-(((x[j] - data1[k])/h - estimated_mean)**2)/(2*estimated_variance))/(sqrt(2*np.pi*estimated_variance))
        
        y[j] = sum1/sum2
    return y

"""auteur: Alexis """   
#mean square error function
def mse(target, estimation):
    n = len(target)
    res = 0
    for k in range(n):
        res += (target[k] - estimation[k])**2
    return res/n

x, y = get_trainable_data(prostate_file)
l = np.arange(-0.5, 6, 0.1)
y_hat = kde(l, y, h=0.1)

plt.plot(l, y_hat)
plt.show()
plt.scatter(x[:,0], y)
plt.show()

y_all = kde_2D(x[:,0], x[:,0], y, h=0.05)
print("mean squared error on whole dataset = " + str(mse(y, y_all)))
plt.scatter(x[:,0], y)
plt.scatter(x[:,0], y_all, c='r')
plt.show()