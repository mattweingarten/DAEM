import numpy as np
import pandas as pd
import os
import re
import warnings

def extract_indices(s):
    l = re.findall("r([0-9]+)_c([0-9]+)", s)
    if len(l)==0:
        raise ValueError(f"Found 0 matches for input: {s}")
    if len(l)>1:
        warnings.warn(f"Found {len(l)} matches, expected 1")
    row, col = l[0]
    return int(row)-1, int(col)-1

def read_and_preprocess(path):
    raw = pd.read_csv(path)
    matrix_col = raw["Id"].map(lambda s : extract_indices(s)[1])
    matrix_row = raw["Id"].map(lambda s : extract_indices(s)[0])
    return pd.DataFrame(
        data={
            'Id' : raw['Id'],
            'matrix_row' : matrix_row,
            'matrix_col' : matrix_col,
            'Prediction' : raw['Prediction']
        }
    )

def normalize(data, by_col=False):
    primary = "matrix_col" if by_col else "matrix_row"
    secondary  = "matrix_row" if by_col else "matrix_col"
    mean = data.groupby(primary).mean()['Prediction']
    std = data.groupby(primary).std()['Prediction'].map(lambda x: max(1e-3, x))
    groups = data.groupby(primary).groups
    normalized = data['Prediction'].copy().rename("Normalized")
    for idx in groups.keys():
        normalized.loc[groups[idx]] -= mean[idx]
        normalized.loc[groups[idx]] *= 1./std[idx]
    return data.join(normalized), mean, std