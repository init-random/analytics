import numpy as np
import sklearn.datasets
from numpy import random
from pandas.io.parsers import read_csv
import os


def simple_linear():
    x = np.array([[1., 2.],
                  [1., 3.],
                  [1., 4.]])
    y = np.array([[5.08],
                  [6.9],
                  [9.1]])
    return x, y


def linear_data(slope=1, intercept=0, n_points=50, noise_var=0.5):
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    for i in range(n_points):
        x[i] = i
        y[i] = slope * i + intercept + random.randn(1) * noise_var
    return x, y


def iris_data():
    return sklearn.datasets.load_iris().data


def diabetes_data_2d():
    d = sklearn.datasets.load_diabetes()
    return d.data[:, 2], d.target


def housing_data():
    root = os.path.dirname(os.path.realpath(__file__))
    path = root + '/raw/housing/x.dat'
    x = read_csv(path, sep=' ', skipinitialspace=True, header=None, names=['living_area', 'bedrooms'])
    path = root + '/raw/housing/y.dat'
    y = read_csv(path, sep=' ', skipinitialspace=True, header=None, names=['price'])
    return x, y


def temperature_data():
    weather_path = './raw/weather/weather.csv'
    df = read_csv(weather_path, sep='\t', engine='c', lineterminator='\n')
    x = df[['Lat']]
    y = df[['JanTemp']]
    return x, y


