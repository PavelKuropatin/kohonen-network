import numpy as np
import pandas as pd


def read_file(path, ext, options=None) -> np.ndarray:
    if options is None:
        options = {}
    if ext == "csv":
        df = pd.read_csv(path, header=None)
    else:
        raise RuntimeError(f"not implemented format: {ext}")
    return np.asarray(df.values)
