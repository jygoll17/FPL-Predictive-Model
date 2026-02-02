"""FPL Points Predictor interface."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.ml.features import FeatureEngineer
from src.ml.model import FPLPointsModel


class FPLPredictor:
    """Interface for making FPL points predictions."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model (defaults to models/fpl_points_model.joblib)
        """
        if model_path is None:
            model_path = MODELS_DIR / "fpl_points_model.joblib"
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n\n"
                "Train the model first with:\n  python train_model.py\n\n"
                "For a quick run (no tuning):\n  python train_model.py --quick"
            )

        self.model = FPLPointsModel.load(str(path))
        self.feature_engineer = FeatureEngineer()

    def predict_gameweek(
        self,
        gameweek: int,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        min_minutes: int = 0,
    ) -> pd.DataFrame:
        """
        Predict points for all players in a gameweek.

        Returns:
            DataFrame with player_id, player_name, predicted_points, and other info
        """
        # Engineer features up to target gameweek
        df = self.feature_engineer.engineer_features(
            gameweek_stats,
            players,
            clubs,
            fixtures,
            target_gw=gameweek,
        )

        # Filter to target gameweek
        df_target = df[df["gameweek"] == gameweek]
        
        # If target gameweek doesn't exist (forward prediction), create it
        if df_target.empty:
            # Get latest gameweek for each player
            max_gw = df["gameweek"].max()
            df_latest = df[df["gameweek"] == max_gw].copy()
            # Create rows for target gameweek with latest features
            df_target = df_latest.copy()
            df_target["gameweek"] = gameweek
        
        df = df_target

        # Filter by minimum minutes if specified
        if min_minutes > 0:
            df = df[df["minutes"] >= min_minutes]

        # Get feature columns
        feature_cols = self.feature_engineer.feature_names
        X = df[feature_cols].fillna(0).values

        # Make predictions
        predictions = self.model.predict(X)

        # Create results DataFrame
        results = pd.DataFrame({
            "player_id": df["player_id"].values,
            "player_name": df.get("web_name", df.get("player_name", "")).values,
            "position": df.get("position", "").values,
            "team": df.get("name_team", "").values,
            "price": df.get("price", 0.0).values,
            "predicted_points": predictions,
        })

        return results.sort_values("predicted_points", ascending=False)

    def predict_best_team(
        self,
        gameweek: int,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        budget: float = 100.0,
    ) -> Dict[str, List]:
        """
        Predict optimal team selection.

        Returns:
            Dictionary with 'team' (15 players) and 'starting_xi' (11 players)
        """
        predictions = self.predict_gameweek(
            gameweek, gameweek_stats, players, clubs, fixtures
        )

        # Simple greedy selection (can be improved with optimization)
        team = []
        budget_remaining = budget
        positions_needed = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

        # Sort by value (points per million)
        predictions["value"] = predictions["predicted_points"] / (
            predictions["price"] + 0.1
        )
        predictions = predictions.sort_values("value", ascending=False)

        for _, player in predictions.iterrows():
            pos = player["position"]
            if positions_needed.get(pos, 0) > 0 and player["price"] <= budget_remaining:
                team.append(player.to_dict())
                positions_needed[pos] -= 1
                budget_remaining -= player["price"]

                if sum(positions_needed.values()) == 0:
                    break

        # Select starting XI (top 11 by predicted points)
        starting_xi = sorted(team, key=lambda x: x["predicted_points"], reverse=True)[:11]

        return {"team": team, "starting_xi": starting_xi}

    def get_transfer_suggestions(
        self,
        gameweek: int,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        current_team: List[int],
        max_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Get transfer suggestions.

        Args:
            current_team: List of current player IDs
            max_price: Maximum price for suggested players

        Returns:
            DataFrame with transfer suggestions
        """
        predictions = self.predict_gameweek(
            gameweek, gameweek_stats, players, clubs, fixtures
        )

        # Filter out current team
        suggestions = predictions[~predictions["player_id"].isin(current_team)]

        if max_price:
            suggestions = suggestions[suggestions["price"] <= max_price]

        suggestions["value"] = suggestions["predicted_points"] / (
            suggestions["price"] + 0.1
        )

        return suggestions.sort_values("predicted_points", ascending=False)
