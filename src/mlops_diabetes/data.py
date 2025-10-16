from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from .version import RANDOM_STATE


FEATURE_ORDER = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]


def load_data(test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ds = load_diabetes(as_frame=True)
    X = ds.frame.drop(columns=["target"])
    y = ds.frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, y_train, X_test, y_test
