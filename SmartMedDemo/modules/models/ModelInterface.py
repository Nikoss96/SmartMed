from abc import ABC

import numpy as np


class Model(ABC):
    
    def __init__(self, x: np.array, y: np.array):
        self.x = x
        self.y = y

    def fit(self):
        self.model = self.model.fit(self.x, self.y)

    def predict(self, x: np.array) -> float:
        return self.model.predict(x)
