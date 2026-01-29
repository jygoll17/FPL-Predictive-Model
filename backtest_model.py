#!/usr/bin/env python3
"""Model backtesting script."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.features import FeatureEngineer
from src.ml.model import FPLPointsModel
from src.ml.training import TrainingPipeline
from src.storage import CSVHandler


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest FPL points prediction model")
    parser.add_argument("--start-gw", type=int, default=10, help="Starting gameweek")
    parser.add_argument("--end-gw", type=int, default=23, help="Ending gameweek")
    parser.add_argument("--quick", action="store_true", help="Quick backtest (no tuning)")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    handler = CSVHandler()
    players_df = pd.DataFrame([p.to_csv_row() for p in handler.load_players()])
    clubs_df = pd.DataFrame([c.to_csv_row() for c in handler.load_clubs()])
    fixtures_df = pd.DataFrame([f.to_csv_row() for f in handler.load_fixtures()])
    gameweek_stats_df = handler.load_gameweek_stats()

    if len(players_df) == 0:
        print("Error: No data found. Run collect_data.py first.")
        return

    print(f"Backtesting from GW{args.start_gw} to GW{args.end_gw}...")

    # Backtest for each gameweek
    results = []
    feature_engineer = FeatureEngineer()

    for gw in range(args.start_gw, args.end_gw + 1):
        print(f"\nBacktesting GW{gw}...")

        # Prepare training data (up to GW-1)
        train_df = feature_engineer.engineer_features(
            gameweek_stats_df,
            players_df,
            clubs_df,
            fixtures_df,
            target_gw=gw - 1,
        )
        train_df = train_df[train_df["minutes"] >= 45]

        # Prepare test data (GW)
        test_df = feature_engineer.engineer_features(
            gameweek_stats_df,
            players_df,
            clubs_df,
            fixtures_df,
            target_gw=gw,
        )
        test_df = test_df[
            (test_df["gameweek"] == gw) & (test_df["minutes"] >= 45)
        ]

        if len(test_df) == 0:
            print(f"  No data for GW{gw}, skipping...")
            continue

        # Get features
        feature_cols = feature_engineer.feature_names
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["points"].values

        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["points"].values

        # Train model (XGBoost only when --quick)
        pipeline = TrainingPipeline(
            use_ensemble=not args.quick,
            use_tuning=False,  # Skip tuning for backtest speed
            n_trials=0,
        )
        pipeline.model = FPLPointsModel(lgbm_params=None) if args.quick else FPLPointsModel()
        pipeline.model.train(X_train, y_train, feature_names=feature_cols)

        # Predict
        y_pred = pipeline.model.predict(X_test)

        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        within_2 = np.mean(np.abs(y_test - y_pred) <= 2)
        within_3 = np.mean(np.abs(y_test - y_pred) <= 3)

        results.append({
            "gameweek": gw,
            "mae": mae,
            "rmse": rmse,
            "within_2": within_2,
            "within_3": within_3,
            "n_samples": len(y_test),
        })

        print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, Within 2: {within_2:.1%}")

    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Backtest Summary:")
    print("=" * 60)
    print(f"Average MAE:        {results_df['mae'].mean():.3f}")
    print(f"Average RMSE:       {results_df['rmse'].mean():.3f}")
    print(f"Average Within 2:   {results_df['within_2'].mean():.1%}")
    print(f"Average Within 3:   {results_df['within_3'].mean():.1%}")
    print(f"\nTotal samples:     {results_df['n_samples'].sum()}")


if __name__ == "__main__":
    main()
