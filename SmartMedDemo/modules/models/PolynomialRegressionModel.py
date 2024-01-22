from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from .BaseModel import BaseModel


class PolynomialRegressionModel(BaseModel):

    def __init__(self, x, y, degree=2, col_idx=1):
        poly = PolynomialFeatures(degree, include_bias=False)
        x_poly = poly.fit_transform(x)
        x_poly1 = x_poly[:, 1 + col_idx]
        print(x_poly.shape)
        super().__init__(LinearRegression, x_poly1.reshape(-1, 1), y)
