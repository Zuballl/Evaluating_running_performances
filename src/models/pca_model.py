from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PCAResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray
    explained_variance_ratio: float
    scaler: MinMaxScaler | None = None


def run_pca(df_train: pd.DataFrame, df_test: pd.DataFrame, df_all: pd.DataFrame) -> tuple[PCAResult, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train)
    scaled_test = scaler.transform(df_test)
    scaled_all = scaler.transform(df_all)

    pca = PCA(n_components=1)
    pca.fit(scaled_train)

    # MSE on unseen test data
    scores_test = pca.transform(scaled_test)
    reconstructed_test = pca.inverse_transform(scores_test)
    mse = float(mean_squared_error(scaled_test, reconstructed_test))

    # Scores for all data (for ranking)
    scores = pca.transform(scaled_all).flatten()

    result = PCAResult(
        name="pca",
        architecture="Input -> PCA(1) -> Output",
        mse=mse,
        scores=scores,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
        scaler=scaler,
    )
    return result, scaler