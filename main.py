import numpy as np
import pandas as pd

"""
auteur : Tom Dauvé
"""
def load_data(filepath):
    df = pd.read_table(filepath)
    data = df.values
    output  = []
    for i in range(len(data)):
        output.append(data[i][1:])
    return output

"""
auteur : Tom Dauvé
"""
def clean_data(data):
    for i in range(len(data)):
        if(data[i][-1] == 'T'):
            data[i][-1] = 1
        else:
            data[i][-1] = 0
    return np.array(data)
    