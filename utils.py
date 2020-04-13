import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def get_test_train_data():
    data = pd.read_csv('to_train_data.csv')
    features = ['Mean', 'STD', 'Sum', 'Null%', 'Age', 'Gender']

    x = preprocessing.scale(data[features].copy())
    y = data['Label'].copy()

    return train_test_split(x, y, test_size=0.03, random_state=0)
