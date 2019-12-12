from collections import defaultdict
from typing import List

import numpy as np


class KohonenNetwork:

    def __init__(self, weights: np.ndarray, names: List[str], data: np.ndarray, learn_speed: float):
        self.__weights = weights
        self.__names = names
        self.__data = data
        self.__normalized_data = ()
        self.__learn_speed = learn_speed
        self.__classes = ()

    @staticmethod
    def __normalize_column(values: np.ndarray):
        max_val = values.max(initial=None)
        min_val = values.min(initial=None)
        return np.asarray([
            (val - min_val) / (max_val - min_val)
            for val in values
        ])

    def normalize(self):
        self.__normalized_data = np.apply_along_axis(self.__normalize_column, 0, self.__data)

    def learn(self, ):
        def __define_class(x):
            r_j = np.array([
                np.sqrt(np.sum(np.subtract(x, w) ** 2))
                for w in self.__weights
            ])
            class_i = r_j.argmin()
            for j, w in enumerate(self.__weights[class_i]):
                self.__weights[class_i][j] = w + self.__learn_speed * (x[j] - w)
            return class_i

        self.__classes = np.apply_along_axis(__define_class, 1, self.__normalized_data)

    def get_result(self):
        result = defaultdict(list)
        for i, _class in enumerate(self.__classes):
            result[_class].append(self.__names[i])
        return result
