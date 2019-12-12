import os

from constants import ASSETS
from fs_utils import read_file
from generate_utils import generate_weights
from network import KohonenNetwork

DATA_FILE = os.path.join(ASSETS, "data.csv")
WEIGHTS = os.path.join(ASSETS, "weights.csv")

if __name__ == "__main__":
    clusters = 3
    features = 7
    learn_speed = 0.3
    learn_step = 0.05

    data = read_file(DATA_FILE, "csv")
    # weights = read_file(WEIGHTS, "csv")
    weights = generate_weights(clusters, features)

    print(data)
    print(weights)

    network = KohonenNetwork(weights, data, learn_speed, learn_step)
    learning_data = data

    clusters = network.learn(learning_data)

    print(clusters)
    print(network.weights)
