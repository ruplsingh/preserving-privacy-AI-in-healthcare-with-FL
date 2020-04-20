import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def get_test_train_data():
    data = pd.read_csv('to_train_data.csv')
    features = ['Mean', 'STD', 'Sum', 'Null%', 'Age', 'Gender']

    x = preprocessing.scale(data[features].copy())
    y = data['Label'].copy()

    return train_test_split(x, y, test_size=0.03, random_state=0)


def get_train_data():
    data = pd.read_csv('to_train_data.csv')
    features = ['Mean', 'STD', 'Sum', 'Null%', 'Gender', 'Age']

    x = data[features].copy().values.tolist()
    y = [[i] for i in data['Label'].copy().values.tolist()]

    return x, y
