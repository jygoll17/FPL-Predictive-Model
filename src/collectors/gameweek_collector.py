"""Gameweek performance data collector."""

import csv
import io
from typing import Dict, List, Optional

from src.config import FPL_BASE_URL, FPL_DATA_CSV_URL, FPL_BOOTSTRAP_URL
from src.models.gameweek_performance import GameweekPerformance

from .base import BaseCollector


class GameweekCollector(BaseCollector):
    """Collects per-gameweek player performance data."""

    async def collect(self) -> List[GameweekPerformance]:
        """Collect gameweek stats from FPL-Data.co.uk CSV or FPL API fallback."""
        # Try CSV first
        performances = await self._collect_from_csv()
        if performances:
            return performances

        # Fallback to API
        return await self._collect_from_api()

    async def _collect_from_csv(self) -> List[GameweekPerformance]:
        """Collect from FPL-Data.co.uk CSV (only if URL returns real CSV with element IDs)."""
        csv_content = await self.fetch_csv(FPL_DATA_CSV_URL)
        if not csv_content:
            return []
        # FPL-Data URL often returns HTML, not CSV; require header with "element" for player ID
        first_line = csv_content.split("\n")[0].lower()
        if "element" not in first_line and "element_id" not in first_line:
            return []  # Fall back to API so we get proper fpl_id

        # Get player and team mappings
        bootstrap_data = await self.fetch_json(FPL_BOOTSTRAP_URL)
        if not bootstrap_data:
            return []

        player_map: Dict[int, str] = {}
        for element in bootstrap_data.get("elements", []):
            player_map[element["id"]] = element.get("web_name", "")

        team_map: Dict[int, str] = {}
        for team in bootstrap_data.get("teams", []):
            team_map[team["id"]] = team["name"]

        performances = []
        reader = csv.DictReader(io.StringIO(csv_content))

        for row in reader:
            try:
                fpl_id = int(row.get("element", 0))
                player_name = player_map.get(fpl_id, row.get("name", ""))
                opponent_id = int(row.get("opponent_team", 0))
                opponent = team_map.get(opponent_id, f"Team {opponent_id}")

                perf = GameweekPerformance(
                    fpl_id=fpl_id,
                    player_name=player_name,
                    gameweek=int(row.get("round", 0)),
                    opponent_team_id=opponent_id,
                    opponent=opponent,
                    home_away="H" if row.get("was_home", "").lower() == "true" else "A",
                    fixture_id=int(row.get("fixture", 0)) if row.get("fixture") else None,
                    kickoff_time=row.get("kickoff_time"),
                    date=row.get("kickoff_time", "").split("T")[0] if row.get("kickoff_time") else None,
                    minutes=int(row.get("minutes", 0)),
                    points=int(row.get("total_points", 0)),
                    was_home=row.get("was_home", "").lower() == "true",
                    goals_scored=int(row.get("goals_scored", 0)),
                    assists=int(row.get("assists", 0)),
                    clean_sheets=int(row.get("clean_sheets", 0)),
                    goals_conceded=int(row.get("goals_conceded", 0)),
                    own_goals=int(row.get("own_goals", 0)),
                    penalties_saved=int(row.get("penalties_saved", 0)),
                    penalties_missed=int(row.get("penalties_missed", 0)),
                    yellow_cards=int(row.get("yellow_cards", 0)),
                    red_cards=int(row.get("red_cards", 0)),
                    saves=int(row.get("saves", 0)),
                    bonus=int(row.get("bonus", 0)),
                    bps=int(row.get("bps", 0)),
                    influence=float(row.get("influence", 0)),
                    creativity=float(row.get("creativity", 0)),
                    threat=float(row.get("threat", 0)),
                    ict_index=float(row.get("ict_index", 0)),
                    expected_goals=float(row.get("expected_goals", 0)),
                    expected_assists=float(row.get("expected_assists", 0)),
                    expected_goal_involvements=float(row.get("expected_goal_involvements", 0)),
                    expected_goals_conceded=float(row.get("expected_goals_conceded", 0)),
                    value=int(row.get("value", 0)) if row.get("value") else None,
                    selected=int(row.get("selected", 0)) if row.get("selected") else None,
                    transfers_in=int(row.get("transfers_in", 0)) if row.get("transfers_in") else None,
                    transfers_out=int(row.get("transfers_out", 0)) if row.get("transfers_out") else None,
                    transfers_balance=int(row.get("transfers_balance", 0)) if row.get("transfers_balance") else None,
                )
                performances.append(perf)
            except (ValueError, KeyError) as e:
                # Skip invalid rows
                continue

        return performances

    async def _collect_from_api(self) -> List[GameweekPerformance]:
        """Collect from FPL API element-summary endpoints."""
        # Get player and team mappings
        bootstrap_data = await self.fetch_json(FPL_BOOTSTRAP_URL)
        if not bootstrap_data:
            return []

        player_map: Dict[int, str] = {}
        for element in bootstrap_data.get("elements", []):
            player_map[element["id"]] = element.get("web_name", "")

        team_map: Dict[int, str] = {}
        for team in bootstrap_data.get("teams", []):
            team_map[team["id"]] = team["name"]

        performances = []
        elements = bootstrap_data.get("elements", [])

        for element in elements:
            fpl_id = element["id"]
            player_name = player_map.get(fpl_id, "")

            url = f"{FPL_BASE_URL}/element-summary/{fpl_id}/"
            data = await self.fetch_json(url)
            if not data:
                continue

            history = data.get("history", [])
            for hist in history:
                perf = GameweekPerformance.from_fpl_api(hist, player_name, team_map)
                performances.append(perf)

        return performances
