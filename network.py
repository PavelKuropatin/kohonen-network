import numpy as np


class KohonenNetwork:

    def __init__(self, weights: np.ndarray, data: np.ndarray, learn_speed: float, learn_step: float):
        self.__weights = weights
        self.__data = data
        self.__learn_speed = learn_speed
        self.__learn_step = learn_step

    @property
    def weights(self):
        return self.__weights

    def learn(self, learning_data: np.ndarray, output=False):

        if len(learning_data.shape) == 1:
            learning_data = np.asarray([learning_data])

        out = []
        learn_speed = self.__learn_speed
        while learn_speed > 0:
            for x in learning_data:
                r_j = np.array([
                    np.sqrt(np.sum(np.subtract(x, w) ** 2))
                    for w in self.__weights
                ])
                cluster = r_j.argmin()
                for j, w in enumerate(self.__weights[cluster]):
                    self.__weights[cluster][j] = w + learn_speed * (x[j] - w)
                learn_speed -= self.__learn_step
                if output:
                    out.append((cluster, x))
        return out

    def classify(self, x: np.ndarray):
        return self.learn(x, output=True)
