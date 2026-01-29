#!/usr/bin/env python3
"""Model analysis and reporting script."""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ANALYSIS_DIR, MODELS_DIR
from src.ml.model import FPLPointsModel
from src.ml.predictor import FPLPredictor
from src.storage import CSVHandler

# Set style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)


def image_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def generate_html_report(
    metrics: dict,
    feature_importance: dict,
    perm_importance: dict,
    predictions_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    images: dict,
    output_dir: Path,
):
    """Generate HTML analysis report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FPL Points Prediction Model Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #37003c;
            border-bottom: 3px solid #37003c;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #00ff87;
            margin-top: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #00ff87;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff87;
        }}
        .metric-label {{
            color: #aaa;
            margin-top: 5px;
        }}
        img {{
            max-width: 100%;
            margin: 20px 0;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2a2a2a;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #444;
        }}
        th {{
            background: #37003c;
            color: #fff;
        }}
        tr:hover {{
            background: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FPL Points Prediction Model Analysis</h1>
        
        <h2>Model Performance Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{metrics['mae']:.3f}</div>
                <div class="metric-label">Mean Absolute Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['rmse']:.3f}</div>
                <div class="metric-label">Root Mean Squared Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['r2']:.3f}</div>
                <div class="metric-label">R² Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['within_2']:.1%}</div>
                <div class="metric-label">Within 2 Points</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['within_3']:.1%}</div>
                <div class="metric-label">Within 3 Points</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['bias']:.3f}</div>
                <div class="metric-label">Bias</div>
            </div>
        </div>

        <h2>Feature Importance (XGBoost)</h2>
        <img src="data:image/png;base64,{images['feature_importance']}" alt="Feature Importance" />

        <h2>Permutation Importance</h2>
        <img src="data:image/png;base64,{images['permutation_importance']}" alt="Permutation Importance" />

        <h2>Prediction vs Actual</h2>
        <img src="data:image/png;base64,{images['prediction_vs_actual']}" alt="Prediction vs Actual" />

        <h2>Error Distribution</h2>
        <img src="data:image/png;base64,{images['error_distribution']}" alt="Error Distribution" />
    </div>
</body>
</html>
"""
    output_path = output_dir / "model_analysis_report.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"✓ HTML report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze FPL points prediction model")
    parser.add_argument("--output", type=str, help="Output directory", default=str(ANALYSIS_DIR))

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading model...")
    model = FPLPointsModel.load(str(MODELS_DIR / "fpl_points_model.joblib"))

    # Load data for analysis
    print("Loading data...")
    handler = CSVHandler()
    players_df = pd.DataFrame([p.to_csv_row() for p in handler.load_players()])
    clubs_df = pd.DataFrame([c.to_csv_row() for c in handler.load_clubs()])
    fixtures_df = pd.DataFrame([f.to_csv_row() for f in handler.load_fixtures()])
    gameweek_stats_df = handler.load_gameweek_stats()

    # Get validation data
    from src.ml.training import TrainingPipeline
    pipeline = TrainingPipeline(use_ensemble=False, use_tuning=False)
    X_train, X_val, y_train, y_val = pipeline.prepare_data(
        gameweek_stats_df, players_df, clubs_df, fixtures_df
    )

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_val)

    # Calculate metrics
    metrics = model._calculate_metrics(y_val, y_pred)

    # Feature importance
    print("Calculating feature importance...")
    feature_importance = model.get_feature_importance()
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    features, importances = zip(*top_features)
    ax.barh(range(len(features)), importances)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importance (XGBoost)")
    ax.invert_yaxis()
    img_feat_imp = image_to_base64(fig)

    # Permutation importance
    print("Calculating permutation importance...")
    perm_result = permutation_importance(
        model.xgb_model, X_val[:1000], y_val[:1000], n_repeats=10, random_state=42, n_jobs=-1
    )
    top_perm = sorted(
        zip(model.feature_names, perm_result.importances_mean),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    features, importances = zip(*top_perm)
    ax.barh(range(len(features)), importances)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Top 20 Permutation Importance")
    ax.invert_yaxis()
    img_perm_imp = image_to_base64(fig)

    # Prediction vs Actual
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hexbin(y_val, y_pred, gridsize=30, cmap="Blues")
    ax.plot([0, 20], [0, 20], "r--", lw=2)
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title("Prediction vs Actual Points")
    img_pred_vs_actual = image_to_base64(fig)

    # Error distribution
    errors = y_pred - y_val
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, edgecolor="black")
    ax.axvline(0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    img_error_dist = image_to_base64(fig)

    # Save feature importance CSV
    feat_df = pd.DataFrame(top_features, columns=["feature", "importance"])
    feat_df.to_csv(output_dir / "feature_importance.csv", index=False)

    perm_df = pd.DataFrame(top_perm, columns=["feature", "permutation_importance"])
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    # Generate HTML report
    images = {
        "feature_importance": img_feat_imp,
        "permutation_importance": img_perm_imp,
        "prediction_vs_actual": img_pred_vs_actual,
        "error_distribution": img_error_dist,
    }

    generate_html_report(
        metrics,
        feature_importance,
        dict(top_perm),
        pd.DataFrame(),
        pd.DataFrame(),
        images,
        output_dir,
    )

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
