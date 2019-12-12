import numpy as np


def generate_weights(clusters: int, features: int) -> np.ndarray:
    return np.random.random((clusters, features))
