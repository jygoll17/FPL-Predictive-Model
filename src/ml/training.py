"""Training pipeline with Optuna hyperparameter tuning."""

import optuna
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from src.ml.features import FeatureEngineer
from src.ml.model import FPLPointsModel


class TrainingPipeline:
    """Training pipeline with hyperparameter tuning."""

    def __init__(
        self,
        use_ensemble: bool = True,
        use_tuning: bool = True,
        n_trials: int = 30,
        timeout: int = 300,
    ):
        """
        Initialize training pipeline.

        Args:
            use_ensemble: Whether to use ensemble (XGBoost + LightGBM)
            use_tuning: Whether to use Optuna for hyperparameter tuning
            n_trials: Number of Optuna trials
            timeout: Timeout for Optuna optimization (seconds)
        """
        self.use_ensemble = use_ensemble
        self.use_tuning = use_tuning
        self.n_trials = n_trials
        self.timeout = timeout
        self.feature_engineer = FeatureEngineer()
        self.model: Optional[FPLPointsModel] = None

    def prepare_data(
        self,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        train_gw_end: int = 18,
        val_gw_start: int = 19,
        val_gw_end: int = 23,
        min_minutes: int = 45,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.

        Returns:
            (X_train, X_val, y_train, y_val)
        """
        # Engineer features for training period
        train_df = self.feature_engineer.engineer_features(
            gameweek_stats,
            players,
            clubs,
            fixtures,
            target_gw=train_gw_end,
        )

        # Engineer features for validation period
        val_df = self.feature_engineer.engineer_features(
            gameweek_stats,
            players,
            clubs,
            fixtures,
            target_gw=val_gw_end,
        )

        # Filter to validation gameweeks
        val_df = val_df[
            (val_df["gameweek"] >= val_gw_start) & (val_df["gameweek"] <= val_gw_end)
        ]

        # Filter by minimum minutes
        train_df = train_df[train_df["minutes"] >= min_minutes]
        val_df = val_df[val_df["minutes"] >= min_minutes]

        # #region agent log
        _log_path = __import__("pathlib").Path(__file__).resolve().parent.parent.parent / ".cursor" / "debug.log"
        with open(_log_path, "a") as _f: _f.write(__import__("json").dumps({"location": "training.prepare_data:after_filter", "message": "train/val sizes", "data": {"train_rows": len(train_df), "val_rows": len(val_df), "train_gw_end": train_gw_end, "val_gw_start": val_gw_start, "val_gw_end": val_gw_end}, "hypothesisId": "D"}) + "\n")
        # #endregion

        # Get feature columns
        feature_cols = self.feature_engineer.feature_names

        # Prepare X and y
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["points"].values

        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df["points"].values

        return X_train, X_val, y_train, y_val

    def compute_sample_weights(
        self, gameweeks: np.ndarray, recency_factor: float = 0.1
    ) -> np.ndarray:
        """Compute sample weights based on recency."""
        max_gw = gameweeks.max()
        normalized_gw = (gameweeks - gameweeks.min()) / (max_gw - gameweeks.min() + 1)
        weights = np.exp(recency_factor * normalized_gw * 5)
        weights = weights / weights.mean()  # Normalize to mean 1
        return weights

    def train(
        self,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        train_gw_end: int = 18,
        val_gw_start: int = 19,
        val_gw_end: int = 23,
        min_minutes: int = 45,
        use_sample_weights: bool = True,
    ) -> Dict[str, float]:
        """
        Train the model.

        Returns:
            Validation metrics
        """
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(
            gameweek_stats,
            players,
            clubs,
            fixtures,
            train_gw_end,
            val_gw_start,
            val_gw_end,
            min_minutes,
        )

        # Get gameweeks for sample weighting
        train_df = self.feature_engineer.engineer_features(
            gameweek_stats,
            players,
            clubs,
            fixtures,
            target_gw=train_gw_end,
        )
        train_df = train_df[train_df["minutes"] >= min_minutes]
        train_gws = train_df["gameweek"].values

        sample_weight = None
        if use_sample_weights:
            sample_weight = self.compute_sample_weights(train_gws)

        # Hyperparameter tuning
        if self.use_tuning:
            best_params = self._tune_hyperparameters(
                X_train, y_train, X_val, y_val, sample_weight
            )
            xgb_params = best_params["xgb_params"]
            lgbm_params = best_params["lgbm_params"] if self.use_ensemble else None
            ensemble_weight = best_params.get("ensemble_weight", 0.6)
        else:
            xgb_params = None
            lgbm_params = None
            ensemble_weight = 0.6

        # Train final model
        self.model = FPLPointsModel(
            xgb_params=xgb_params,
            lgbm_params=lgbm_params if self.use_ensemble else None,
            ensemble_weight=ensemble_weight,
        )

        metrics = self.model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            sample_weight=sample_weight,
            feature_names=self.feature_engineer.feature_names,
        )

        return metrics

    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """Tune hyperparameters using Optuna."""

        def objective(trial):
            # XGBoost parameters
            xgb_params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 500),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 0.95),
                "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
                "reg_alpha": trial.suggest_float("xgb_alpha", 0.01, 1.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_lambda", 0.1, 2.0),
                "random_state": 42,
                "n_jobs": -1,
            }

            if self.use_ensemble:
                # LightGBM parameters
                lgbm_params = {
                    "n_estimators": trial.suggest_int("lgbm_n_estimators", 200, 500),
                    "max_depth": trial.suggest_int("lgbm_max_depth", 3, 7),
                    "learning_rate": trial.suggest_float("lgbm_lr", 0.01, 0.1, log=True),
                    "subsample": trial.suggest_float("lgbm_subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.6, 0.95),
                    "min_child_samples": trial.suggest_int("lgbm_min_child", 10, 50),
                    "reg_alpha": trial.suggest_float("lgbm_alpha", 0.01, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("lgbm_lambda", 0.1, 2.0),
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }

                ensemble_weight = trial.suggest_float("ensemble_weight", 0.3, 0.7)
            else:
                lgbm_params = None
                ensemble_weight = 1.0

            # Train model
            model = FPLPointsModel(
                xgb_params=xgb_params,
                lgbm_params=lgbm_params,
                ensemble_weight=ensemble_weight,
            )
            model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)

            # Evaluate
            y_pred = model.predict(X_val)
            mae = np.mean(np.abs(y_val - y_pred))

            return mae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Extract best parameters
        best_params = study.best_params
        xgb_params = {
            "n_estimators": best_params["xgb_n_estimators"],
            "max_depth": best_params["xgb_max_depth"],
            "learning_rate": best_params["xgb_lr"],
            "subsample": best_params["xgb_subsample"],
            "colsample_bytree": best_params["xgb_colsample"],
            "min_child_weight": best_params["xgb_min_child"],
            "reg_alpha": best_params["xgb_alpha"],
            "reg_lambda": best_params["xgb_lambda"],
            "random_state": 42,
            "n_jobs": -1,
        }

        result = {"xgb_params": xgb_params}

        if self.use_ensemble:
            lgbm_params = {
                "n_estimators": best_params["lgbm_n_estimators"],
                "max_depth": best_params["lgbm_max_depth"],
                "learning_rate": best_params["lgbm_lr"],
                "subsample": best_params["lgbm_subsample"],
                "colsample_bytree": best_params["lgbm_colsample"],
                "min_child_samples": best_params["lgbm_min_child"],
                "reg_alpha": best_params["lgbm_alpha"],
                "reg_lambda": best_params["lgbm_lambda"],
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            result["lgbm_params"] = lgbm_params
            result["ensemble_weight"] = best_params["ensemble_weight"]

        return result
