"""CSV storage handler with ID mapping management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import (
    CLUBS_CSV,
    FIXTURE_HISTORY_CSV,
    ID_MAPPING_JSON,
    METADATA_JSON,
    PLAYER_GAMEWEEK_CSV,
    PLAYERS_CSV,
)
from src.models.club import Club
from src.models.fixture import Fixture
from src.models.gameweek_performance import GameweekPerformance
from src.models.player import Player


class CSVHandler:
    """Handles CSV file I/O and ID mappings."""

    def __init__(self):
        """Initialize handler."""
        self.id_mappings: Dict[str, Dict[int, int]] = {
            "players": {},
            "clubs": {},
            "fixtures": {},
        }

    def load_id_mappings(self) -> None:
        """Load ID mappings from JSON file."""
        if ID_MAPPING_JSON.exists():
            with open(ID_MAPPING_JSON, "r") as f:
                self.id_mappings = json.load(f)

    def save_id_mappings(self) -> None:
        """Save ID mappings to JSON file."""
        with open(ID_MAPPING_JSON, "w") as f:
            json.dump(self.id_mappings, f, indent=2)

    def update_metadata(self, data_type: str) -> None:
        """Update metadata with last update timestamp."""
        metadata = {}
        if METADATA_JSON.exists():
            with open(METADATA_JSON, "r") as f:
                metadata = json.load(f)

        metadata[data_type] = datetime.now().isoformat()

        with open(METADATA_JSON, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_players(self, players: List[Player]) -> None:
        """Save players to CSV with ID mapping."""
        self.load_id_mappings()

        rows = []
        for player in players:
            if player.player_id is None:
                # Assign sequential ID
                if player.fpl_id not in self.id_mappings["players"]:
                    next_id = len(self.id_mappings["players"]) + 1
                    self.id_mappings["players"][player.fpl_id] = next_id
                player.player_id = self.id_mappings["players"][player.fpl_id]

            rows.append(player.to_csv_row())

        df = pd.DataFrame(rows)
        df.to_csv(PLAYERS_CSV, index=False)
        self.save_id_mappings()
        self.update_metadata("players")

    def load_players(self) -> List[Player]:
        """Load players from CSV."""
        if not PLAYERS_CSV.exists():
            return []

        df = pd.read_csv(PLAYERS_CSV)
        players = []
        for _, row in df.iterrows():
            player_dict = row.to_dict()
            # Convert NaN to None
            player_dict = {k: (None if pd.isna(v) else v) for k, v in player_dict.items()}
            players.append(Player(**player_dict))

        return players

    def save_clubs(self, clubs: List[Club]) -> None:
        """Save clubs to CSV with ID mapping."""
        self.load_id_mappings()

        rows = []
        for club in clubs:
            if club.club_id is None:
                if club.fpl_id not in self.id_mappings["clubs"]:
                    next_id = len(self.id_mappings["clubs"]) + 1
                    self.id_mappings["clubs"][club.fpl_id] = next_id
                club.club_id = self.id_mappings["clubs"][club.fpl_id]

            rows.append(club.to_csv_row())

        df = pd.DataFrame(rows)
        df.to_csv(CLUBS_CSV, index=False)
        self.save_id_mappings()
        self.update_metadata("clubs")

    def load_clubs(self) -> List[Club]:
        """Load clubs from CSV."""
        if not CLUBS_CSV.exists():
            return []

        df = pd.read_csv(CLUBS_CSV)
        clubs = []
        for _, row in df.iterrows():
            club_dict = row.to_dict()
            club_dict = {k: (None if pd.isna(v) else v) for k, v in club_dict.items()}
            clubs.append(Club(**club_dict))

        return clubs

    def save_fixtures(self, fixtures: List[Fixture]) -> None:
        """Save fixtures to CSV with ID mapping."""
        self.load_id_mappings()

        rows = []
        for fixture in fixtures:
            if fixture.fixture_id is None:
                if fixture.fpl_id not in self.id_mappings["fixtures"]:
                    next_id = len(self.id_mappings["fixtures"]) + 1
                    self.id_mappings["fixtures"][fixture.fpl_id] = next_id
                fixture.fixture_id = self.id_mappings["fixtures"][fixture.fpl_id]

            rows.append(fixture.to_csv_row())

        df = pd.DataFrame(rows)
        df.to_csv(FIXTURE_HISTORY_CSV, index=False)
        self.save_id_mappings()
        self.update_metadata("fixtures")

    def load_fixtures(self) -> List[Fixture]:
        """Load fixtures from CSV."""
        if not FIXTURE_HISTORY_CSV.exists():
            return []

        df = pd.read_csv(FIXTURE_HISTORY_CSV)
        fixtures = []
        for _, row in df.iterrows():
            fixture_dict = row.to_dict()
            fixture_dict = {k: (None if pd.isna(v) else v) for k, v in fixture_dict.items()}
            fixtures.append(Fixture(**fixture_dict))

        return fixtures

    def save_gameweek_stats(
        self, performances: List[GameweekPerformance], player_map: Dict[int, int]
    ) -> None:
        """Save gameweek stats to CSV with player ID mapping."""
        # If id_mappings has no players (e.g. fresh or JSON keys), build map from players CSV
        if not player_map and PLAYERS_CSV.exists():
            players_df = pd.read_csv(PLAYERS_CSV)
            if "fpl_id" in players_df.columns and "player_id" in players_df.columns:
                for _, row in players_df.iterrows():
                    try:
                        fid, pid = int(row["fpl_id"]), int(row["player_id"])
                        player_map[fid] = pid
                        player_map[str(fid)] = pid
                    except (ValueError, TypeError):
                        pass

        rows = []
        record_id = 1

        for perf in performances:
            # Map FPL ID to our sequential ID (JSON keys are strings)
            perf.player_id = player_map.get(perf.fpl_id) or player_map.get(str(perf.fpl_id))
            perf.record_id = record_id
            record_id += 1
            rows.append(perf.to_csv_row())

        if rows:
            df = pd.DataFrame(rows)
            # Append if file exists, otherwise create new
            if PLAYER_GAMEWEEK_CSV.exists():
                existing_df = pd.read_csv(PLAYER_GAMEWEEK_CSV)
                df = pd.concat([existing_df, df], ignore_index=True)
                # Remove duplicates
                df = df.drop_duplicates(
                    subset=["player_id", "gameweek"], keep="last"
                )

            df.to_csv(PLAYER_GAMEWEEK_CSV, index=False)
            self.update_metadata("gameweek_stats")

    def load_gameweek_stats(self) -> pd.DataFrame:
        """Load gameweek stats as DataFrame. Ensures player_id exists for feature engineering."""
        # #region agent log
        _log_path = "/Users/jaywanthgollakarum/Documents/GitHub/FPL-Predictive-Model/.cursor/debug.log"
        with open(_log_path, "a") as _f: _f.write(json.dumps({"location": "csv_handler.load_gameweek_stats:entry", "message": "load_gameweek_stats", "data": {"file_exists": PLAYER_GAMEWEEK_CSV.exists()}, "hypothesisId": "A"}) + "\n")
        # #endregion
        if not PLAYER_GAMEWEEK_CSV.exists():
            return pd.DataFrame()

        df = pd.read_csv(PLAYER_GAMEWEEK_CSV)
        # #region agent log
        with open(_log_path, "a") as _f: _f.write(json.dumps({"location": "csv_handler.load_gameweek_stats:after_read", "message": "after read_csv", "data": {"shape": list(df.shape), "columns": list(df.columns), "has_fpl_id": "fpl_id" in df.columns, "has_player_id": "player_id" in df.columns}, "hypothesisId": "A"}) + "\n")
        # #endregion

        # Normalize FPL-Data CSV column name to fpl_id (external CSV uses "element")
        if "fpl_id" not in df.columns and "element" in df.columns:
            df["fpl_id"] = pd.to_numeric(df["element"], errors="coerce")

        # Ensure player_id exists (required by feature engineer). Derive from fpl_id if missing.
        if "player_id" not in df.columns and "fpl_id" in df.columns:
            self.load_id_mappings()
            player_map = self.id_mappings.get("players", {})
            # If id_mappings is empty, build fpl_id -> player_id from players CSV
            if not player_map and PLAYERS_CSV.exists():
                players_df = pd.read_csv(PLAYERS_CSV)
                if "fpl_id" in players_df.columns and "player_id" in players_df.columns:
                    for _, row in players_df.iterrows():
                        try:
                            fid, pid = int(row["fpl_id"]), int(row["player_id"])
                            player_map[fid] = pid
                            player_map[str(fid)] = pid
                        except (ValueError, TypeError):
                            pass

            def fpl_to_player_id(fpl_id):
                if pd.isna(fpl_id):
                    return None
                try:
                    fid = int(fpl_id)
                    return player_map.get(fid) or player_map.get(str(fid))
                except (ValueError, TypeError):
                    return None

            mapped = df["fpl_id"].map(fpl_to_player_id)
            df["player_id"] = pd.to_numeric(mapped, errors="coerce")

        # If player_id exists but has NaNs, fill from fpl_id mapping
        if "player_id" in df.columns and "fpl_id" in df.columns and df["player_id"].isna().any():
            self.load_id_mappings()
            player_map = self.id_mappings.get("players", {})
            if not player_map and PLAYERS_CSV.exists():
                players_df = pd.read_csv(PLAYERS_CSV)
                if "fpl_id" in players_df.columns and "player_id" in players_df.columns:
                    for _, row in players_df.iterrows():
                        try:
                            fid, pid = int(row["fpl_id"]), int(row["player_id"])
                            player_map[fid] = pid
                            player_map[str(fid)] = pid
                        except (ValueError, TypeError):
                            pass

            def fpl_to_player_id(fpl_id):
                if pd.isna(fpl_id):
                    return None
                try:
                    fid = int(fpl_id)
                    return player_map.get(fid) or player_map.get(str(fid))
                except (ValueError, TypeError):
                    return None

            missing = df["player_id"].isna()
            mapped = df.loc[missing, "fpl_id"].map(fpl_to_player_id)
            df.loc[missing, "player_id"] = pd.to_numeric(mapped, errors="coerce")

        # Ensure player_id is int64 for merge (drop rows with missing player_id)
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
            _before_drop = len(df)
            df = df.dropna(subset=["player_id"])
            df["player_id"] = df["player_id"].astype("int64")
            # #region agent log
            with open(_log_path, "a") as _f: _f.write(json.dumps({"location": "csv_handler.load_gameweek_stats:exit", "message": "before return", "data": {"rows_before_drop": _before_drop, "rows_after_drop": len(df), "player_id_dtype": str(df["player_id"].dtype)}, "hypothesisId": "A"}) + "\n")
            # #endregion
        else:
            # #region agent log
            with open(_log_path, "a") as _f: _f.write(json.dumps({"location": "csv_handler.load_gameweek_stats:exit_no_player_id", "message": "no player_id column", "data": {"rows": len(df), "columns": list(df.columns)[:10]}, "hypothesisId": "A"}) + "\n")
            # #endregion

        return df

    def get_player_id_map(self) -> Dict[int, int]:
        """Get mapping from FPL ID to sequential player ID."""
        self.load_id_mappings()
        return self.id_mappings.get("players", {})

    def get_club_id_map(self) -> Dict[int, int]:
        """Get mapping from FPL ID to sequential club ID."""
        self.load_id_mappings()
        return self.id_mappings.get("clubs", {})
