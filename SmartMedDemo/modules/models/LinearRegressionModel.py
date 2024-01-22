# import numpy as np
# import sklearn.metrics as sm
# from scipy import stats
# import pandas as pd

from sklearn.linear_model import LinearRegression

from .BaseModel import BaseModel


class LinearRegressionModel(BaseModel):

    def __init__(self, x, y):
        super().__init__(LinearRegression, x, y)
