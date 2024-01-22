import numpy as np

from .models import *
from .models.TreeModel import TreeModel


class ModelChooseException(Exception):
    pass


class ModelManipulator():

    def __init__(self, model_type: str, x: np.array, y: np.array, extra_param: np.array = None):

        if model_type == 'linreg':
            self.model = LinearRegressionModel(x, y)

        elif model_type == 'logreg':
            self.model = LogisticRegressionModel(x, y)

        elif model_type == 'polyreg':
            self.model = PolynomialRegressionModel(x, y)

        elif model_type == 'tree':
            self.model = TreeModel(x, y, extra_param)
        else:
            raise ModelChooseException

    def create(self):
        return self.model
