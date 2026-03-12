from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .features import FEATURE_COLUMNS, generate_synthetic_demand_history


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "user_demand_rf_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synthetic user-demand forecasting models.")
    parser.add_argument("--train-end-date", type=str, default="2026-03-11")
    parser.add_argument("--num-train-days", type=int, default=180)
    parser.add_argument("--num-valid-days", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dataset", type=str, default="")
    return parser.parse_args()


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_model(
    train_df: pd.DataFrame,
    target: str,
    *,
    n_estimators: int,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> RandomForestRegressor:
    ## we use a RF to learn the relationship between the features and P_fix, P_flex
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df[target])
    return model


def main() -> None:
    args = parse_args()
    train_end_date = _parse_date(args.train_end_date)
    train_start_date = train_end_date - timedelta(days=args.num_train_days - 1)
    valid_start_date = train_end_date + timedelta(days=1)

    ## generate synthetic demand history for training and validation
    train_df = generate_synthetic_demand_history(
        start_date=train_start_date,
        num_days=args.num_train_days,
        base_seed=args.base_seed,
    )
    ## we generate a separate validation set with a different seed to simulate the generalization performance of the model on unseen data.
    valid_df = generate_synthetic_demand_history(
        start_date=valid_start_date,
        num_days=args.num_valid_days,
        base_seed=args.base_seed + args.num_train_days,
    )

    ## Train separate models for fixed and flexible demand
    pfix_model = _fit_model(
        train_df,
        "Pfixe",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.base_seed,
        n_jobs=args.n_jobs,
    )
    pflex_model = _fit_model(
        train_df,
        "Pflex",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.base_seed + 1,
        n_jobs=args.n_jobs,
    )

    ## Evaluate on validation set
    pfix_pred = pfix_model.predict(valid_df[FEATURE_COLUMNS])
    pflex_pred = pflex_model.predict(valid_df[FEATURE_COLUMNS])

    ## Save the models and features in a single bundle file for easy loading by the forecasting agent.
    metrics = {
        "pfix": _compute_metrics(valid_df["Pfixe"], pfix_pred),
        "pflex": _compute_metrics(valid_df["Pflex"], pflex_pred),
    }

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pfix_model": pfix_model,
        "pflex_model": pflex_model,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": {
            "train_start_date": train_start_date.isoformat(),
            "train_end_date": train_end_date.isoformat(),
            "num_train_days": args.num_train_days,
            "num_valid_days": args.num_valid_days,
            "metrics": metrics,
        },
    }
    joblib.dump(bundle, output_model)

    ## we save the generated dataset as well if an output path is provided
    if args.output_dataset:
        output_dataset = Path(args.output_dataset)
        output_dataset.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                train_df.assign(split="train"),
                valid_df.assign(split="valid"),
            ],
            ignore_index=True,
        ).to_csv(output_dataset, index=False)

    print(f"Saved model bundle to: {output_model}")
    print("Validation metrics:")
    for target, target_metrics in metrics.items():
        print(f"  {target}:")
        for name, value in target_metrics.items():
            print(f"    {name}: {value:.6f}")


if __name__ == "__main__":
    main()
