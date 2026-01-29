#!/usr/bin/env python3
"""Model training script."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR
from src.ml.training import TrainingPipeline
from src.storage import CSVHandler


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train FPL points prediction model")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--no-ensemble", action="store_true", help="Use XGBoost only")
    parser.add_argument("--quick", action="store_true", help="Quick training (no tune, no ensemble)")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--cross-validate", action="store_true", help="Run cross-validation")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    handler = CSVHandler()
    players_df = pd.DataFrame([p.to_csv_row() for p in handler.load_players()])
    clubs_df = pd.DataFrame([c.to_csv_row() for c in handler.load_clubs()])
    fixtures_df = pd.DataFrame([f.to_csv_row() for f in handler.load_fixtures()])
    gameweek_stats_df = handler.load_gameweek_stats()

    if len(players_df) == 0 or len(clubs_df) == 0:
        print("Error: No data found. Run collect_data.py first.")
        return

    print(f"Loaded {len(players_df)} players, {len(clubs_df)} clubs, "
          f"{len(fixtures_df)} fixtures, {len(gameweek_stats_df)} gameweek stats")

    # Configure training
    use_tuning = not (args.no_tune or args.quick)
    use_ensemble = not (args.no_ensemble or args.quick)
    n_trials = args.trials if use_tuning else 0

    if args.cross_validate:
        # Run cross-validation instead of single train/val
        print("\nRunning cross-validation...")
        pipeline = TrainingPipeline(
            use_ensemble=use_ensemble,
            use_tuning=False,
            n_trials=0,
        )
        # Time-based CV: 3 folds (train end GW 14/17/20, val GW 15-17/18-20/21-23)
        cv_splits = [(14, 15, 17), (17, 18, 20), (20, 21, 23)]
        cv_maes = []
        for i, (train_end, val_start, val_end) in enumerate(cv_splits, 1):
            m = pipeline.train(
                gameweek_stats_df, players_df, clubs_df, fixtures_df,
                train_gw_end=train_end, val_gw_start=val_start, val_gw_end=val_end,
            )
            cv_maes.append(m["mae"])
            print(f"  Fold {i} (train GW1-{train_end}, val GW{val_start}-{val_end}): MAE={m['mae']:.3f}")
        print(f"\nCV Mean MAE: {sum(cv_maes) / len(cv_maes):.3f}")
        # Train final model on default split and save
        pipeline.train(gameweek_stats_df, players_df, clubs_df, fixtures_df)
        pipeline.model.save(str(MODELS_DIR / "fpl_points_model.joblib"))
        print(f"✓ Model saved to {MODELS_DIR / 'fpl_points_model.joblib'}")
        return

    print(f"\nTraining configuration:")
    print(f"  Ensemble: {use_ensemble}")
    print(f"  Hyperparameter tuning: {use_tuning}")
    if use_tuning:
        print(f"  Optuna trials: {n_trials}")

    # Train
    pipeline = TrainingPipeline(
        use_ensemble=use_ensemble,
        use_tuning=use_tuning,
        n_trials=n_trials,
    )

    print("\nTraining model...")
    metrics = pipeline.train(
        gameweek_stats_df,
        players_df,
        clubs_df,
        fixtures_df,
    )

    # Save model
    model_path = MODELS_DIR / "fpl_points_model.joblib"
    pipeline.model.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}")

    # Print metrics
    print("\nValidation Metrics:")
    print("=" * 50)
    print(f"MAE:              {metrics['mae']:.3f}")
    print(f"RMSE:             {metrics['rmse']:.3f}")
    print(f"R²:                {metrics['r2']:.3f}")
    print(f"Within 1 point:   {metrics['within_1']:.1%}")
    print(f"Within 2 points:  {metrics['within_2']:.1%}")
    print(f"Within 3 points:  {metrics['within_3']:.1%}")
    print(f"Within 5 points:  {metrics['within_5']:.1%}")
    print(f"Bias:             {metrics['bias']:.3f}")


if __name__ == "__main__":
    main()
