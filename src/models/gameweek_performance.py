"""Gameweek performance data model."""

from typing import Optional
from pydantic import BaseModel, Field


class GameweekPerformance(BaseModel):
    """Individual player performance in a specific gameweek."""

    # IDs
    record_id: Optional[int] = None  # Our sequential ID
    player_id: Optional[int] = None  # Our sequential player ID
    fpl_id: int = Field(..., description="FPL element ID")
    player_name: str = Field(..., description="Player name")

    # Gameweek info
    gameweek: int = Field(..., description="Gameweek number")
    opponent_team_id: int = Field(..., description="Opponent FPL team ID")
    opponent: str = Field(..., description="Opponent team name")
    home_away: str = Field(..., description="H or A")
    fixture_id: Optional[int] = None

    # Timing
    kickoff_time: Optional[str] = None
    date: Optional[str] = None

    # Performance
    minutes: int = Field(default=0)
    points: int = Field(default=0)
    was_home: bool = Field(default=False)

    # Stats
    goals_scored: int = Field(default=0)
    assists: int = Field(default=0)
    clean_sheets: int = Field(default=0)
    goals_conceded: int = Field(default=0)
    own_goals: int = Field(default=0)
    penalties_saved: int = Field(default=0)
    penalties_missed: int = Field(default=0)
    yellow_cards: int = Field(default=0)
    red_cards: int = Field(default=0)
    saves: int = Field(default=0)
    bonus: int = Field(default=0)
    bps: int = Field(default=0)

    # ICT
    influence: float = Field(default=0.0)
    creativity: float = Field(default=0.0)
    threat: float = Field(default=0.0)
    ict_index: float = Field(default=0.0)

    # Expected stats
    expected_goals: float = Field(default=0.0)
    expected_assists: float = Field(default=0.0)
    expected_goal_involvements: float = Field(default=0.0)
    expected_goals_conceded: float = Field(default=0.0)

    # Value and transfers
    value: Optional[int] = None
    selected: Optional[int] = None
    transfers_in: Optional[int] = None
    transfers_out: Optional[int] = None
    transfers_balance: Optional[int] = None

    @classmethod
    def from_fpl_api(
        cls, history: dict, player_name: str, opponent_map: dict
    ) -> "GameweekPerformance":
        """Create GameweekPerformance from FPL API history entry."""
        opponent_id = history.get("opponent_team", 0)
        was_home = history.get("was_home", False)

        return cls(
            fpl_id=history.get("element", 0),
            player_name=player_name,
            gameweek=history.get("round", 0),
            opponent_team_id=opponent_id,
            opponent=opponent_map.get(opponent_id, f"Team {opponent_id}"),
            home_away="H" if was_home else "A",
            fixture_id=history.get("fixture"),
            kickoff_time=history.get("kickoff_time"),
            date=history.get("kickoff_time", "").split("T")[0] if history.get("kickoff_time") else None,
            minutes=history.get("minutes", 0),
            points=history.get("total_points", 0),
            was_home=was_home,
            goals_scored=history.get("goals_scored", 0),
            assists=history.get("assists", 0),
            clean_sheets=history.get("clean_sheets", 0),
            goals_conceded=history.get("goals_conceded", 0),
            own_goals=history.get("own_goals", 0),
            penalties_saved=history.get("penalties_saved", 0),
            penalties_missed=history.get("penalties_missed", 0),
            yellow_cards=history.get("yellow_cards", 0),
            red_cards=history.get("red_cards", 0),
            saves=history.get("saves", 0),
            bonus=history.get("bonus", 0),
            bps=history.get("bps", 0),
            influence=float(history.get("influence", 0)),
            creativity=float(history.get("creativity", 0)),
            threat=float(history.get("threat", 0)),
            ict_index=float(history.get("ict_index", 0)),
            expected_goals=float(history.get("expected_goals", 0)),
            expected_assists=float(history.get("expected_assists", 0)),
            expected_goal_involvements=float(history.get("expected_goal_involvements", 0)),
            expected_goals_conceded=float(history.get("expected_goals_conceded", 0)),
            value=history.get("value"),
            selected=history.get("selected"),
            transfers_in=history.get("transfers_in"),
            transfers_out=history.get("transfers_out"),
            transfers_balance=history.get("transfers_balance"),
        )

    def to_csv_row(self) -> dict:
        """Convert to CSV row dictionary."""
        return self.model_dump(exclude_none=True, by_alias=False)
