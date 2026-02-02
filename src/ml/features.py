"""Feature engineering for FPL points prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class FeatureEngineer:
    """Engineers 73 features from raw FPL data."""

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names: List[str] = []

    def engineer_features(
        self,
        gameweek_stats: pd.DataFrame,
        players: pd.DataFrame,
        clubs: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gw: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Engineer all features for prediction.

        Args:
            gameweek_stats: DataFrame with player_gameweek_stats
            players: DataFrame with player data
            clubs: DataFrame with club data
            fixtures: DataFrame with fixture data
            target_gw: Target gameweek (if None, uses all data)

        Returns:
            DataFrame with engineered features
        """
        # Merge data (suffixes so gameweek_stats columns keep names for rolling/features)
        df = gameweek_stats.copy()
        df = df.merge(
            players,
            left_on="player_id",
            right_on="player_id",
            how="left",
            suffixes=("", "_player"),
        )
        df["opponent_team_id"] = pd.to_numeric(df["opponent_team_id"], errors="coerce")
        df = df.merge(
            clubs,
            left_on="opponent_team_id",
            right_on="fpl_id",
            how="left",
            suffixes=("", "_opponent"),
        )
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")
        # Merge team data for player's team
        df = df.merge(
            clubs,
            left_on="team_id",
            right_on="fpl_id",
            how="left",
            suffixes=("", "_team"),
        )
        df["fixture_id"] = pd.to_numeric(df["fixture_id"], errors="coerce")
        # Merge fixture data
        df = df.merge(
            fixtures,
            left_on="fixture_id",
            right_on="fpl_id",
            how="left",
            suffixes=("", "_fixture"),
        )

        # Sort by player and gameweek
        df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)
        # Filter to target gameweek if specified
        if target_gw is not None:
            df = df[df["gameweek"] <= target_gw]

        # Group by player for rolling calculations
        grouped = df.groupby("player_id")

        # 1. Rolling Averages (Multiple Windows: 3, 5, 10 games)
        df = self._add_rolling_averages(df, grouped)
        # 2. Rolling Sums
        df = self._add_rolling_sums(df, grouped)

        # 3. Momentum/Trend Features
        df = self._add_momentum_features(df, grouped)

        # 4. Consistency Features
        df = self._add_consistency_features(df, grouped)

        # 5. Rate Features
        df = self._add_rate_features(df, grouped)

        # 6. Weighted Averages
        df = self._add_weighted_averages(df, grouped)

        # 7. Opponent Features
        df = self._add_opponent_features(df, clubs, fixtures)
        # 8. Team Strength Features
        df = self._add_team_strength_features(df)

        # 9. Player Features
        df = self._add_player_features(df)

        # 10. Fixture Features
        df = self._add_fixture_features(df)
        # 11. Position Features (One-Hot)
        df = self._add_position_features(df)

        # 12. Position-Specific Features
        df = self._add_position_specific_features(df)

        # Store feature names
        # Exclude IDs, targets, and any string columns (names etc.) so X is numeric for XGBoost
        exclude = [
            "player_id",
            "fpl_id",
            "gameweek",
            "points",
            "record_id",
            "player_name",
            "name",
            "first_name",
            "second_name",
            "web_name",
            "opponent",
            "home_away",
            "kickoff_time",
            "date",
            "short_name",
            "home_team_name",
            "away_team_name",
            "season",
            "status",
            "position",  # raw string; one-hot creates position_GKP, position_DEF, etc.
            "news",
            "news_added",
            "short_name_opponent",
            "short_name_team",
            "name_opponent",
            "name_team",
        ]
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]
        self.feature_names = feature_cols

        return df

    def _add_rolling_averages(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add rolling average features."""
        windows = [3, 5, 10]

        for window in windows:
            # Points
            df[f"avg_points_last{window}"] = (
                grouped["points"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

            # Expected stats
            df[f"avg_xg_last{window}"] = (
                grouped["expected_goals"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            df[f"avg_xa_last{window}"] = (
                grouped["expected_assists"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            df[f"avg_xgi_last{window}"] = (
                grouped["expected_goal_involvements"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            df[f"avg_ict_last{window}"] = (
                grouped["ict_index"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

            # ICT components
            df[f"avg_influence_last{window}"] = (
                grouped["influence"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            df[f"avg_creativity_last{window}"] = (
                grouped["creativity"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )
            df[f"avg_threat_last{window}"] = (
                grouped["threat"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

            # BPS
            df[f"avg_bps_last{window}"] = (
                grouped["bps"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

            # Minutes
            df[f"avg_minutes_last{window}"] = (
                grouped["minutes"]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

        # Overall average (all time)
        df["avg_points"] = (
            grouped["points"].transform(lambda x: x.shift(1).expanding().mean())
        )
        df["avg_xg"] = (
            grouped["expected_goals"].transform(lambda x: x.shift(1).expanding().mean())
        )
        df["avg_xa"] = (
            grouped["expected_assists"].transform(lambda x: x.shift(1).expanding().mean())
        )
        df["avg_xgi"] = (
            grouped["expected_goal_involvements"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        df["avg_ict"] = (
            grouped["ict_index"].transform(lambda x: x.shift(1).expanding().mean())
        )

        return df

    def _add_rolling_sums(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add rolling sum features."""
        window = 5

        df["goals_last_5"] = (
            grouped["goals_scored"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df["assists_last_5"] = (
            grouped["assists"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df["cs_last_5"] = (
            grouped["clean_sheets"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df["bonus_last_5"] = (
            grouped["bonus"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )
        df["games_last_5"] = (
            grouped["minutes"]
            .transform(
                lambda x: (x.shift(1) >= 45).rolling(window=window, min_periods=1).sum()
            )
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add momentum/trend features."""
        # Points momentum: recent 3 games avg - older 3 games avg
        recent_3 = (
            grouped["points"]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )
        older_3 = (
            grouped["points"]
            .transform(
                lambda x: x.shift(4).rolling(window=3, min_periods=1).mean()
            )
        )
        df["points_momentum"] = recent_3 - older_3.fillna(0)

        # xG momentum
        recent_xg = (
            grouped["expected_goals"]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )
        older_xg = (
            grouped["expected_goals"]
            .transform(lambda x: x.shift(4).rolling(window=3, min_periods=1).mean())
        )
        df["xg_momentum"] = recent_xg - older_xg.fillna(0)

        # Form trajectory (derivative)
        df["form_trajectory"] = (
            grouped["points"]
            .transform(lambda x: x.shift(1).diff().rolling(window=3, min_periods=1).mean())
        )

        return df

    def _add_consistency_features(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add consistency features."""
        df["points_std"] = (
            grouped["points"]
            .transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())
        )
        df["points_cv"] = (
            grouped["points"]
            .transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).std()
                / (x.shift(1).rolling(window=10, min_periods=3).mean() + 0.1)
            )
        )
        df["xg_std"] = (
            grouped["expected_goals"]
            .transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())
        )

        return df

    def _add_rate_features(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add rate features."""
        window = 10

        # Haul rate (10+ points)
        hauls = (
            grouped["points"]
            .transform(lambda x: (x.shift(1) >= 10).rolling(window=window, min_periods=1).sum())
        )
        games = (
            grouped["points"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).count())
        )
        df["haul_rate"] = hauls / (games + 0.1)

        # Blank rate (2 or fewer points)
        blanks = (
            grouped["points"]
            .transform(lambda x: (x.shift(1) <= 2).rolling(window=window, min_periods=1).sum())
        )
        df["blank_rate"] = blanks / (games + 0.1)

        # Full 90 rate
        full_90s = (
            grouped["minutes"]
            .transform(lambda x: (x.shift(1) >= 85).rolling(window=window, min_periods=1).sum())
        )
        df["full_90_rate"] = full_90s / (games + 0.1)

        # Goal rate
        goals = (
            grouped["goals_scored"]
            .transform(lambda x: (x.shift(1) > 0).rolling(window=window, min_periods=1).sum())
        )
        df["goal_rate"] = goals / (games + 0.1)

        # Assist rate
        assists = (
            grouped["assists"]
            .transform(lambda x: (x.shift(1) > 0).rolling(window=window, min_periods=1).sum())
        )
        df["assist_rate"] = assists / (games + 0.1)

        # Bonus rate
        bonus_games = (
            grouped["bonus"]
            .transform(lambda x: (x.shift(1) > 0).rolling(window=window, min_periods=1).sum())
        )
        df["bonus_rate"] = bonus_games / (games + 0.1)

        return df

    def _add_weighted_averages(self, df: pd.DataFrame, grouped) -> pd.DataFrame:
        """Add exponentially weighted averages."""
        # Recent games weighted more heavily
        df["weighted_avg_points"] = (
            grouped["points"]
            .transform(
                lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
            )
        )
        df["weighted_avg_xgi"] = (
            grouped["expected_goal_involvements"]
            .transform(
                lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
            )
        )

        return df

    def _add_opponent_features(
        self, df: pd.DataFrame, clubs: pd.DataFrame, fixtures: pd.DataFrame
    ) -> pd.DataFrame:
        """Add opponent-related features."""
        # Opponent goals conceded average (last 5 games)
        opponent_gc = []
        for _, row in df.iterrows():
            opp_id = row.get("opponent_team_id")
            gw = row.get("gameweek")
            if pd.isna(opp_id) or pd.isna(gw):
                opponent_gc.append(0.0)
                continue

            # Get opponent's last 5 games before this gameweek
            opp_fixtures = fixtures[
                ((fixtures["home_team_id"] == opp_id) | (fixtures["away_team_id"] == opp_id))
                & (fixtures["gameweek"] < gw)
                & (fixtures["finished"] == True)
            ].tail(5)

            if len(opp_fixtures) == 0:
                opponent_gc.append(0.0)
                continue

            gc_sum = 0
            for _, fix in opp_fixtures.iterrows():
                if fix["home_team_id"] == opp_id:
                    gc_sum += fix.get("away_score", 0)
                else:
                    gc_sum += fix.get("home_score", 0)

            opponent_gc.append(gc_sum / len(opp_fixtures))

        df["opponent_goals_conceded_avg"] = opponent_gc

        # Opponent clean sheet rate
        opponent_cs_rate = []
        for _, row in df.iterrows():
            opp_id = row.get("opponent_team_id")
            gw = row.get("gameweek")
            if pd.isna(opp_id) or pd.isna(gw):
                opponent_cs_rate.append(0.0)
                continue

            opp_fixtures = fixtures[
                ((fixtures["home_team_id"] == opp_id) | (fixtures["away_team_id"] == opp_id))
                & (fixtures["gameweek"] < gw)
                & (fixtures["finished"] == True)
            ].tail(10)

            if len(opp_fixtures) == 0:
                opponent_cs_rate.append(0.0)
                continue

            cs_count = 0
            for _, fix in opp_fixtures.iterrows():
                if fix["home_team_id"] == opp_id:
                    if fix.get("away_score", 0) == 0:
                        cs_count += 1
                else:
                    if fix.get("home_score", 0) == 0:
                        cs_count += 1

            opponent_cs_rate.append(cs_count / len(opp_fixtures))

        df["opponent_cs_rate"] = opponent_cs_rate

        # Opponent xGA (from club stats)
        df["opponent_xga"] = df.get("xga", 0.0).fillna(0.0)

        # Defensive weakness composite
        df["opponent_defensive_weakness"] = (
            df["opponent_goals_conceded_avg"] * 0.4
            + (1 - df["opponent_cs_rate"]) * 0.3
            + df["opponent_xga"] * 0.3
        )

        return df

    def _add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team strength features."""
        # Player's team strength (from merged team data)
        df["strength_attack_home"] = df.get("strength_attack_home_team", pd.Series(0, index=df.index)).fillna(0)
        df["strength_attack_away"] = df.get("strength_attack_away_team", pd.Series(0, index=df.index)).fillna(0)
        df["strength_defence_home"] = df.get("strength_defence_home_team", pd.Series(0, index=df.index)).fillna(0)
        df["strength_defence_away"] = df.get("strength_defence_away_team", pd.Series(0, index=df.index)).fillna(0)
        df["strength_overall_home"] = df.get("strength_overall_home_team", pd.Series(0, index=df.index)).fillna(0)
        df["strength_overall_away"] = df.get("strength_overall_away_team", pd.Series(0, index=df.index)).fillna(0)

        # Opponent strength (from opponent merge)
        df["opponent_strength_attack"] = df.get("strength_attack_home", pd.Series(0, index=df.index)).fillna(0)
        df["opponent_strength_defence"] = df.get("strength_defence_home", pd.Series(0, index=df.index)).fillna(0)
        df["opponent_strength_overall"] = df.get("strength_overall_home", pd.Series(0, index=df.index)).fillna(0)

        # Attack vs Defence mismatch
        is_home = df.get("was_home", pd.Series(False, index=df.index)).fillna(False)
        attack_strength = df["strength_attack_home"].where(
            is_home, df["strength_attack_away"]
        )
        defence_strength = df["opponent_strength_defence"]
        df["attack_vs_defence"] = attack_strength - defence_strength

        return df

    def _add_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player-level features."""
        df["price"] = df.get("price", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["form"] = df.get("form", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["selected_by_percent"] = df.get("selected_by_percent", pd.Series(0.0, index=df.index)).fillna(0.0)

        # Ownership tier (1-5)
        df["ownership_tier"] = pd.cut(
            df["selected_by_percent"],
            bins=[0, 1, 5, 10, 25, 100],
            labels=[1, 2, 3, 4, 5],
        ).astype(float).fillna(1.0)

        # Points per million
        total_points = df.get("total_points", pd.Series(0, index=df.index)).fillna(0)
        df["points_per_million"] = total_points / (df["price"] + 0.1)

        return df

    def _add_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fixture-level features."""
        df["is_home"] = df.get("was_home", pd.Series(False, index=df.index)).fillna(False).astype(int)
        df["fixture_difficulty"] = df.get("home_difficulty", pd.Series(0, index=df.index)).fillna(0)
        # If away, use away_difficulty
        away_mask = df["is_home"] == 0
        if away_mask.any():
            away_difficulty = df.loc[away_mask].get("away_difficulty", pd.Series(0, index=df.loc[away_mask].index)).fillna(0)
            df.loc[away_mask, "fixture_difficulty"] = away_difficulty

        return df

    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encoded position features."""
        position = df.get("position", "").fillna("")
        df["pos_GKP"] = (position == "GKP").astype(int)
        df["pos_DEF"] = (position == "DEF").astype(int)
        df["pos_MID"] = (position == "MID").astype(int)
        df["pos_FWD"] = (position == "FWD").astype(int)

        return df

    def _add_position_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific features."""
        position = df.get("position", "").fillna("")

        # Clean sheet opportunity (DEF/GKP)
        cs_rate = df.get("opponent_cs_rate", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["cs_opportunity"] = (
            ((position == "DEF") | (position == "GKP")).astype(int)
            * (1 - cs_rate)
        )

        # Attacking opportunity
        avg_xg = df.get("avg_xg_last5", pd.Series(0.0, index=df.index)).fillna(0.0)
        def_weakness = df.get("opponent_defensive_weakness", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["attacking_opportunity"] = avg_xg * def_weakness

        # Weighted returns (position-weighted goals/assists)
        pos_fwd = df.get("pos_FWD", pd.Series(0, index=df.index))
        pos_mid = df.get("pos_MID", pd.Series(0, index=df.index))
        pos_def = df.get("pos_DEF", pd.Series(0, index=df.index))
        goal_weight = pos_fwd * 1.0 + pos_mid * 0.7 + pos_def * 0.5
        assist_weight = pos_mid * 1.0 + pos_fwd * 0.8 + pos_def * 0.6
        goals_last_5 = df.get("goals_last_5", pd.Series(0, index=df.index)).fillna(0)
        assists_last_5 = df.get("assists_last_5", pd.Series(0, index=df.index)).fillna(0)
        df["weighted_returns"] = goal_weight * goals_last_5 + assist_weight * assists_last_5

        return df
