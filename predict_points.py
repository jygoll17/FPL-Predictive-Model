#!/usr/bin/env python3
"""Points prediction script."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.check_omp import check_ml_backends

check_ml_backends()

from src.ml.predictor import FPLPredictor
from src.storage import CSVHandler


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predict FPL points for a gameweek")
    parser.add_argument("gameweek", type=int, help="Gameweek number")
    parser.add_argument("--top", type=int, help="Show top N players only")
    parser.add_argument("--position", choices=["GKP", "DEF", "MID", "FWD"], help="Filter by position")
    parser.add_argument("--max-price", type=float, help="Maximum price")
    parser.add_argument("--output", help="Output CSV file path")

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

    # Load model and predict
    print(f"Predicting points for Gameweek {args.gameweek}...")
    predictor = FPLPredictor()

    predictions = predictor.predict_gameweek(
        args.gameweek,
        gameweek_stats_df,
        players_df,
        clubs_df,
        fixtures_df,
    )

    # Apply filters
    if args.position:
        predictions = predictions[predictions["position"] == args.position]

    if args.max_price:
        predictions = predictions[predictions["price"] <= args.max_price]

    if args.top:
        predictions = predictions.head(args.top)

    # Display results
    print(f"\nTop {len(predictions)} Predictions for GW{args.gameweek}:")
    print("=" * 80)
    print(f"{'Player':<25} {'Pos':<4} {'Team':<20} {'Price':<6} {'Points':<6}")
    print("-" * 80)

    for _, row in predictions.iterrows():
        print(
            f"{row['player_name']:<25} {row['position']:<4} "
            f"{row['team']:<20} {row['price']:<6.1f} {row['predicted_points']:<6.2f}"
        )

    # Save to CSV if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n✓ Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
