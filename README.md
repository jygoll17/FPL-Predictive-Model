# FPL Points Prediction System

A machine learning system that predicts Fantasy Premier League player points for upcoming gameweeks. The system collects data from multiple sources, engineers 73 features, and trains an XGBoost + LightGBM ensemble model to achieve ~2.4 MAE (Mean Absolute Error).

## Features

- **Data Collection**: Automated collection from FPL API, Premier League website, and FPL-Data.co.uk
- **Feature Engineering**: 73 engineered features including rolling averages, momentum, consistency metrics, and position-specific features
- **Ensemble Model**: XGBoost + LightGBM ensemble with Optuna hyperparameter tuning
- **Prediction System**: Predict points for any gameweek with transfer suggestions and team optimization
- **Analysis & Reporting**: Comprehensive model analysis with HTML reports and visualizations

## Installation

### Prerequisites

- macOS (tested on darwin 21.5.0+)
- Python 3.12+
- Poetry 2.0+

### Setup

```bash
# Install Python (if not already installed)
brew install python@3.14

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# For macOS: Install OpenMP (required for XGBoost)
brew install libomp

# Set library path for XGBoost
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Install project dependencies
poetry install
```

## Usage

### 1. Collect Data

```bash
# Collect all data
poetry run python collect_data.py

# Collect specific data types
poetry run python collect_data.py --players
poetry run python collect_data.py --clubs
poetry run python collect_data.py --fixtures
poetry run python collect_data.py --gameweeks

# Check data status
poetry run python collect_data.py --status
```

### 2. Train Model

```bash
# Full training with hyperparameter tuning
poetry run python train_model.py

# Quick training (no tuning, no ensemble)
poetry run python train_model.py --quick

# Custom options
poetry run python train_model.py --no-tune --no-ensemble
poetry run python train_model.py --trials 50
```

### 3. Make Predictions

```bash
# Predict points for a gameweek
poetry run python predict_points.py 24

# Filter options
poetry run python predict_points.py 24 --top 20
poetry run python predict_points.py 24 --position MID
poetry run python predict_points.py 24 --max-price 8.0

# Export to CSV
poetry run python predict_points.py 24 --output predictions.csv
```

### 4. Backtest Model

```bash
# Default backtest (GW 10-23)
poetry run python backtest_model.py

# Custom range
poetry run python backtest_model.py --start-gw 12 --end-gw 20

# Quick backtest
poetry run python backtest_model.py --quick
```

### 5. Analyze Model

```bash
# Generate full analysis report
poetry run python analyze_model.py

# Custom output directory
poetry run python analyze_model.py --output ./my_analysis
```

## Project Structure

```
fpl-predictor/
├── pyproject.toml          # Project dependencies
├── collect_data.py          # Data collection script
├── train_model.py           # Model training script
├── predict_points.py        # Prediction script
├── backtest_model.py        # Backtesting script
├── analyze_model.py         # Analysis script
│
├── src/                     # Source code
│   ├── config.py            # Configuration constants
│   ├── models/              # Pydantic data models
│   ├── collectors/          # Data collectors
│   ├── storage/             # CSV handlers
│   └── ml/                  # ML components
│       ├── features.py     # Feature engineering
│       ├── model.py        # XGBoost + LightGBM model
│       ├── training.py     # Training pipeline
│       └── predictor.py     # Prediction interface
│
├── data/                    # Data storage
├── models/                  # Trained models
└── analysis/                # Analysis outputs
```

## Model Performance

Target metrics (from specification):
- **Mean Absolute Error (MAE)**: ~2.41 points
- **Within 2 points**: 52.2%
- **Within 3 points**: 73.2%
- **Within 5 points**: 90.3%
- **Model Bias**: < 0.1 points

## Data Sources

- **FPL API**: `https://fantasy.premierleague.com/api`
- **Premier League Website**: `https://www.premierleague.com/stats`
- **FPL-Data.co.uk**: `https://www.fpl-data.co.uk/statistics`

## Notes

1. **Data Freshness**: Run `collect_data.py` weekly to update data
2. **OpenMP**: Required on macOS for XGBoost (`brew install libomp`)
3. **Memory**: Training uses ~2GB RAM
4. **Time**: Full pipeline (collect + train + analyze) takes ~10 minutes
5. **API Rate Limits**: Collector respects 0.5s delay between requests

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
