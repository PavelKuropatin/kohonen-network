import numpy as np


def generate_weights(clusters_cnt: int, features_cnt: int) -> np.ndarray:
    a = 0.5 - 1 / np.sqrt(features_cnt)
    b = 0.5 + 1 / np.sqrt(features_cnt)
    return np.random.uniform(a, b, (clusters_cnt, features_cnt))
