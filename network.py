from collections import defaultdict

import numpy as np


class KohonenNetwork:

    def __init__(self, weights: np.ndarray, data: np.ndarray, learn_speed: float, learn_step: float):
        self.__weights = weights
        self.__data = data
        self.__learn_speed = learn_speed
        self.__learn_step = learn_step
        self.__clusters = defaultdict(list)

    @property
    def clusters(self):
        return self.__clusters

    @property
    def weights(self):
        return self.__weights

    def learn(self, learning_data: np.ndarray):

        if len(learning_data.shape) == 1:
            learning_data = np.asarray([learning_data])

        learn_speed = self.__learn_speed
        while learn_speed > 0:
            for x_i, x in enumerate(learning_data):
                r_j = np.array([
                    np.sqrt(np.sum(np.subtract(x, w) ** 2))
                    for w in self.__weights
                ])
                req_r = r_j.argmin()
                self.__clusters[req_r].append(x_i)
                for j, w in enumerate(self.__weights[req_r]):
                    self.__weights[req_r][j] = w + learn_speed * (x[j] - w)
                learn_speed -= self.__learn_step
