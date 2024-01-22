from sklearn.linear_model import LogisticRegression

from .BaseModel import BaseModel


class LogisticRegressionModel(BaseModel):

    def __init__(self, x, y):
        super().__init__(LogisticRegression, x, y)
