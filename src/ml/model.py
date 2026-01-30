"""FPL Points Prediction Model (XGBoost + LightGBM Ensemble)."""

import joblib
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LightGBM is optional; on macOS it requires libomp (brew install libomp)
try:
    import lightgbm as lgb
except OSError:
    lgb = None  # e.g. libomp not installed


class FPLPointsModel:
    """XGBoost + LightGBM ensemble model for FPL points prediction."""

    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
        ensemble_weight: float = 0.6,
    ):
        """
        Initialize model.

        Args:
            xgb_params: XGBoost parameters
            lgbm_params: LightGBM parameters
            ensemble_weight: Weight for XGBoost (1 - weight for LightGBM)
        """
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.lgbm_model: Optional[Any] = None
        self.feature_names: List[str] = []
        # Disable LightGBM if libomp not available (e.g. macOS without brew install libomp)
        if lgbm_params is not None and lgb is None:
            warnings.warn(
                "LightGBM could not be loaded (e.g. libomp missing on macOS). "
                "Using XGBoost only. Install with: brew install libomp",
                UserWarning,
                stacklevel=2,
            )
            lgbm_params = None
        self.ensemble_weight = ensemble_weight if lgbm_params is not None else 1.0

        # Default parameters (XGBoost always; LightGBM only when ensemble)
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        # None when XGBoost-only (no ensemble)
        self.lgbm_params = lgbm_params
        if self.lgbm_params is None:
            self.ensemble_weight = 1.0  # XGBoost only
        else:
            # Use provided LightGBM params; default if empty dict
            self.lgbm_params = lgbm_params or {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the ensemble model.

        Returns:
            Dictionary with training metrics
        """
        if feature_names:
            self.feature_names = feature_names

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.xgb_model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
        )

        # Train LightGBM only when using ensemble and LightGBM is available
        if self.lgbm_params is not None and lgb is not None:
            self.lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.lgbm_model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                if X_val is not None
                else None,
            )
        else:
            self.lgbm_model = None

        # Evaluate on validation set if provided
        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble or XGBoost only."""
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")

        xgb_pred = self.xgb_model.predict(X)

        if self.lgbm_model is not None:
            lgbm_pred = self.lgbm_model.predict(X)
            ensemble_pred = (
                self.ensemble_weight * xgb_pred
                + (1 - self.ensemble_weight) * lgbm_pred
            )
        else:
            ensemble_pred = xgb_pred

        # Clip to reasonable range [0, 20]
        ensemble_pred = np.clip(ensemble_pred, 0, 20)

        return ensemble_pred

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Within X points accuracy
        within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
        within_2 = np.mean(np.abs(y_true - y_pred) <= 2)
        within_3 = np.mean(np.abs(y_true - y_pred) <= 3)
        within_5 = np.mean(np.abs(y_true - y_pred) <= 5)

        # Bias
        bias = np.mean(y_pred - y_true)

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "within_1": within_1,
            "within_2": within_2,
            "within_3": within_3,
            "within_5": within_5,
            "bias": bias,
        }

    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """Get feature importance from XGBoost model."""
        if self.xgb_model is None:
            return {}

        importance = self.xgb_model.get_booster().get_score(importance_type=importance_type)
        if self.feature_names:
            # Map feature names
            feature_importance = {}
            for i, name in enumerate(self.feature_names):
                feature_key = f"f{i}"
                if feature_key in importance:
                    feature_importance[name] = importance[feature_key]
            return feature_importance

        return importance

    def save(self, path: str) -> None:
        """Save model to file."""
        model_data = {
            "xgb_model": self.xgb_model,
            "lgbm_model": self.lgbm_model,
            "feature_names": self.feature_names,
            "xgb_params": self.xgb_params,
            "lgbm_params": self.lgbm_params,
            "ensemble_weight": self.ensemble_weight,
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "FPLPointsModel":
        """Load model from file."""
        model_data = joblib.load(path)
        model = cls(
            xgb_params=model_data["xgb_params"],
            lgbm_params=model_data.get("lgbm_params"),
            ensemble_weight=model_data.get("ensemble_weight", 1.0),
        )
        model.xgb_model = model_data["xgb_model"]
        model.lgbm_model = model_data.get("lgbm_model")
        model.feature_names = model_data.get("feature_names", [])
        return model
