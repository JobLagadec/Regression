import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, tree, neural_network
from math import *
import pandas as pda
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

housing_file = "data_regression/HousingData.csv"
prostate_file = "data_regression/prostate.data"

"""auteur: Alexis"""

def load_csv(file_name):
    csv_data = pda.read_csv(file_name)
    data = csv_data.values
    targets = data[:,-1]
    data = data[:,:-1]
    labels = list(csv_data.columns)
    return data, targets, labels

"""auteurs: Alexis, Corentin """
def mean(column):
    l = len(column)
    mean = 0
    for k in range(l):
        if(np.isnan(column[k]) == False):
            mean += column[k]
        else: #we should not consider the unknown elements
            l-=1
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

"""auteur: Alexis Guermont"""   
def normalize_data(data):
    res = data
    n, m = data.shape
    maxes = np.zeros(m)
    for k in range(m):
        maxes[k] = max(data[:,k])
        res[:,k] = res[:,k]/maxes[k]
    return res

"""
auteur : Tom Dauve
"""
def load_data_data(filepath):
    df = pda.read_table(filepath)
    data = df.values
    output  = data[:,1:9]
    #for i in range(len(data)):
        #output.append(data[i][1:len(data[i])-1])
    y = data[:,9]
    labels = list(df.columns)[1:10]
    return np.array(output), np.array(y), labels

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

"""auteur: Alexis """ 
def get_trainable_data(file):
    if (".data" in file):
        cleaned_prostate_data, y,labels = load_data_data(file)
        normalized_prostate_data = normalize_data(cleaned_prostate_data)
        x = normalized_prostate_data
    elif(".csv" in file):
        housing_data, y, labels = load_csv(file)
        cleaned_housing_data = clean_csv_data(housing_data)
        x = normalize_data(cleaned_housing_data)
    else:
        return ("file format not supported yet by this code, our engineers are currently working on it")
    
    return (x,y,labels)


""" auteur: Pierre-Adrien """
""" sépare la base donnée en apprentissage et test en utilisant le ratio donnée"""
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

""" auteur: Pierre-Adrien """
""" affiche toutes les comosantes 2 par deux"""
def Matrix_Plot(x_data,y_data,labels):
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation 
    data = np.concatenate((x_data,y_data),axis = 1)
    data = data.astype(np.float)
    xy_df = pda.DataFrame(data, columns=labels)
    pda.plotting.scatter_matrix(xy_df, alpha=1, figsize=(15, 15))
    plt.show()
    
def Correlation_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation 
    data = np.concatenate((x_data,y_data),axis = 1)
    data = data.astype(np.float)
    return np.corrcoef(np.transpose(data))

def Covariance_Matrix(x_data, y_data):
    y_data = np.reshape(y_data, (len(y_data),1)) # necessaire pour concatenation  
    data = np.concatenate((x_data,y_data),axis = 1)
    data = data.astype(np.float)
    return np.cov(np.transpose(data))


"""auteur : Tom Dauve"""
# PCA pour réduire le nombre de composants
def PCA_function(data, targets, N_components = 2 ):
    pca = PCA(n_components = N_components)
    pc = pca.fit_transform(data, targets)
    plt.scatter(pc[:,0], targets)
    plt.title('target function of PC1')
    plt.show()
    if N_components == 2 :
        ax = plt.axes(projection='3d')
        ax.scatter3D(pc[:,0], pc[:,1], targets)
        plt.show()
    return pc


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

"""Auteur: Alexis"""
#This function takes input x and target y as arguments, from the correlation
#matrix it takes the feature the most correlated to y and apply kde estimation on it
def train_kde(x, y, h=0):
    M = Correlation_Matrix(x ,y)[-1]
    index = 0
    max_coeff = 0
    for k in range(len(M)-1):
        if (abs(M[k])>max_coeff):
            max_coeff = abs(M[k])
            index = k
    x_train_set , y_train_set , x_test_set , y_test_set = get_train_test_sets(x, y, train_ratio = 0.75)
    x_train = x_train_set[:,k]
    x_test = x_test_set[:,k]
    y_estimated = kde_2D(x_test, x_train, y_train_set, h)
    test_error = mse(y_test_set, y_estimated)
    return(y_estimated, test_error)
    
"""auteur: Alexis """   
#mean square error function
def mse(target, estimation):
    n = len(target)
    res = 0
    for k in range(n):
        res += (target[k] - estimation[k])**2
    return res/n

def test(prostate_file):
    x, y, _ = get_trainable_data(prostate_file)
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
    return 0

""" Pierre-Adrien """
""" renvoie mse pour toutes les méthodes de régression"""
def Get_MSE(x, y):  # pas utilisé
    x_train_set , y_train_set , x_test_set , y_test_set = get_train_test_sets(x, y, train_ratio = 0.75)
    MSE = 6 * [0]
    
    _, _, _, prediction = Least_Squares_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
    MSE[0] = mse(y_test_set, prediction)
    _, _, _, prediction = Ridge_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
    MSE[1] = mse(y_test_set, prediction)
    _, _, _, prediction = Lasso_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
    MSE[2] = mse(y_test_set, prediction)
    _, prediction = SVM_Regression(x_train_set , y_train_set , x_test_set , y_test_set,kernel="linear")
    MSE[3] = mse(y_test_set, prediction)
    _, prediction = Tree_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
    MSE[4] = mse(y_test_set, prediction)
    _, prediction = NN_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
    MSE[5] = mse(y_test_set, prediction)
    
    return MSE

""" Corentin et Pierre-Adrien """
""" Interface utiliseur : permet de choisir la base de donnée, des représentation graphique, matrice de covariance, 
                        l'utilisation ou non de sélection de composants (PCA), choix de la méthode de régression
"""
def run():
    inpu = ""
    while inpu != "housing" and inpu != "prostate" and inpu != "exit" : # choix de la base de donnée
        inpu = input("Choisir une base de donnée ('housing' ou 'prostate') ou 'exit': \n")
    if inpu == "exit":
        return "END"
    elif inpu == "housing" :
        file = "data_regression/HousingData.csv"
    elif inpu == "prostate":
        file = "data_regression/prostate.data"
        
    x, y, labels = get_trainable_data(file) # extrait et traite les données de la base de données

    
    inpu = ""
    while inpu != "Correlation_Matrix" and inpu != "pass" : # choix d'afficher la matrice de corrélation ou non
        inpu = input("Matrice de correlation ('Correlation_Matrix' ou 'pass') : \n")
    if inpu == "Correlation_Matrix":
        print(Correlation_Matrix(x, y))
    elif inpu == "pass" :
        pass
    
    inpu = ""
    while inpu != "Matrix_Plot" and inpu != "pass" :# choix d'afficher une représentation 2 à 2 de tous les composants
        inpu = input("Représentation des composants 2 par 2 ('Matrix_Plot' ou 'pass') : \n")
    if inpu == "Matrix_Plot":
        Matrix_Plot(x, y, labels)
    elif inpu == "pass" :
        pass
    

    feature_selection = ""
    mse_header = "MSE "
    while feature_selection != "PCA" and feature_selection != "None" : # utilisation ou non de la PCA
        feature_selection = input("Sélection de composants ('PCA' ou 'None') : \n")
        if feature_selection == "PCA":
            x_feature_selection = PCA_function(x, y, 'mle') # mle -> programme choisit le nombre automatiquement de composants à garder
            x_feature_selection = normalize_data(x_feature_selection)
            mse_header = mse_header + "PCA "
        else :
            x_feature_selection = x
    
    x_train_set , y_train_set , x_test_set , y_test_set = get_train_test_sets(x_feature_selection, y, train_ratio = 0.75) # création test/train set
    
    inpu = ""
    while inpu != "exit": # boucle pour pouvoir afficher successivement les résultats de différentes méthodes de régression
        inpu = ""
        while inpu != "1" and inpu != "2" and inpu != "3" and inpu != "4" and inpu != "5" and inpu != "6" and inpu != "7" and inpu != "exit": # choix méthodes de régression
            if inpu == "help":
                inpu = input("no_model : 1 -> least square regression\n\t\t 2 -> ridge with cross validation\n\t\t 3 -> lasso with cross validation\n\t\t 4 -> SVM regression\n\t\t 5 -> tree regression\n\t\t 6 -> neural network regression\n\t\t 7 -> kernel density estimation\n")
            else:
                inpu = input("Choose your model (n° : 1 - 7), or type 'help' for more information or 'exit' :\n")
        
        if inpu == "1":
            _, _, _, y_hat = Least_Squares_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "Least_Squares_Regression "
        elif inpu == "2":
            _, _, _, y_hat = Ridge_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "Ridge_with_CrossVa_Regression "
        elif inpu == "3":
            _, _, _, y_hat = Lasso_with_CrossVa_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "Lasso_with_CrossVa_Regression "
        elif inpu == "4":
            _, y_hat = SVM_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "SVM_Regression "
        elif inpu == "5":
            _, y_hat = Tree_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "Tree_Regression "
        elif inpu == "6":
            _, y_hat = NN_Regression(x_train_set , y_train_set , x_test_set , y_test_set)
            mse_res = mse_header + "NN_Regression "
            
        elif inpu == "7":
            y_hat,_ = train_kde(x_feature_selection , y, h=0.025)
            mse_res = mse_header + "Kernel density estimator "
        
        if inpu != "exit" :
            mse_res = mse_res + str(mse(y_test_set, y_hat))
            print(mse_res)
    
    return "END"

run()