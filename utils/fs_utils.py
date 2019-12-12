import numpy as np
import pandas as pd


def __read_csv(path, read_options=None) -> pd.DataFrame:
    if read_options is None:
        read_options = {}
    skip_header = read_options["skip_header"] if "skip_header" in read_options else False
    pd_options = {}
    if skip_header:
        pd_options["header"] = None
    return pd.read_csv(path, **pd_options)


def read_file(path, read_options, data_mappings=None) -> np.ndarray:
    if data_mappings is None:
        data_mappings = {}

    ext = read_options["format"]
    if ext == "csv":
        df = __read_csv(path, read_options)
    else:
        raise RuntimeError(f"not implemented format: {ext}")

    for key, mappings in data_mappings.items():
        key = df.columns[key] if isinstance(key, int) else key
        df[key] = df[key].apply(lambda x: mappings[x])

    return np.asarray(df.values)
