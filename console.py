import os
from collections import defaultdict
from pprint import pprint

from network import KohonenNetwork
from utils.fs_utils import read_file
from utils.generate_utils import generate_weights

ASSETS = os.path.join(os.getcwd(), "assets")
DATA_FILE = os.path.join(ASSETS, "data.csv")

READ_OPTIONS = defaultdict(str, {
    "format": "csv",
    "skip_header": False
})

DATA_MAPPINGS = {
    "Sex": defaultdict(lambda: -1, {"M": 1, "F": 0}),
    "Credits": defaultdict(lambda: -1, {"Y": 1, "N": 0}),
}

if __name__ == "__main__":
    clusters = 3
    features = 8
    learn_speed = 0.8

    data = read_file(DATA_FILE, READ_OPTIONS, DATA_MAPPINGS)
    names, data = data[:, 0], data[:, 1:]

    weights = generate_weights(clusters, features)

    network = KohonenNetwork(weights, names, data, learn_speed)
    network.normalize()
    network.learn()
    result = network.get_result()
    pprint(result)
