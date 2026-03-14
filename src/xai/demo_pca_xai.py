import argparse
import json

import pandas as pd

from src.data.read_data import PROCESSED_DATA_PATH
from src.xai.pca_xai import explain_sample, fit_pca_explainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo PCA XAI for one sample.")
    parser.add_argument(
        "--clean-data-path",
        type=str,
        default=str(PROCESSED_DATA_PATH),
        help="Path to clean_data.csv used as PCA/XAI reference population.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Index of row from clean_data.csv to explain when --sample-json is not provided.",
    )
    parser.add_argument(
        "--sample-json",
        type=str,
        default=None,
        help=(
            "JSON object with feature values, e.g. "
            "'{\"average_hr\":150,\"average_speed\":11,...}'."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top feature contributions to print.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="standard_clip",
        choices=["standard_clip", "minmax"],
        help="Preprocessing mode used before PCA.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Print response as JSON (backend-friendly).",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indentation for JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.clean_data_path)
    explainer = fit_pca_explainer(df, preprocessing=args.preprocessing)

    if args.sample_json is not None:
        sample_data = json.loads(args.sample_json)
        result = explain_sample(explainer, sample_data)
    else:
        if args.row_index < 0 or args.row_index >= len(df):
            raise ValueError(f"row-index out of range: {args.row_index}, dataset has {len(df)} rows")
        result = explain_sample(explainer, df.iloc[args.row_index])

    top_contributions_df = result.contributions[
        ["feature", "contribution", "contribution_pct_abs"]
    ].head(args.top_k)

    if args.output_json:
        payload = {
            "score": result.score,
            "percentile": result.percentile,
            "top_contributions": top_contributions_df.to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=args.json_indent))
        return

    print(f"score={result.score:.6f}")
    print(f"percentile={result.percentile:.2f}")
    print("top_contributions:")
    print(top_contributions_df.to_string(index=False))


if __name__ == "__main__":
    main()
