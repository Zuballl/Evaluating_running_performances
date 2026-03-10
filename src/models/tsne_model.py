import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def run_tsne(df_numeric: pd.DataFrame, perplexity: int = 30):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    tsne_1d = TSNE(
        n_components=1,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    score_1d = tsne_1d.fit_transform(scaled_data).flatten()

    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding_2d = tsne_2d.fit_transform(scaled_data)

    output_df = df_numeric.copy()
    output_df["tsne_score"] = score_1d
    output_df["x"] = embedding_2d[:, 0]
    output_df["y"] = embedding_2d[:, 1]

    return output_df
