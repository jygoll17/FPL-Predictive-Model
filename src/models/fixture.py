"""Fixture data model."""

from typing import Optional
from pydantic import BaseModel, Field


class Fixture(BaseModel):
    """Fixture model representing a match."""

    # IDs
    fixture_id: Optional[int] = None  # Our sequential ID
    fpl_id: int = Field(..., description="FPL fixture ID")
    home_team_id: int = Field(..., description="Home team FPL ID")
    away_team_id: int = Field(..., description="Away team FPL ID")
    home_team_name: str = Field(..., description="Home team name")
    away_team_name: str = Field(..., description="Away team name")

    # Gameweek and timing
    gameweek: int = Field(..., description="Gameweek number")
    kickoff_time: Optional[str] = None
    date: Optional[str] = None
    season: Optional[str] = None

    # Status
    finished: bool = Field(default=False)
    finished_provisional: bool = Field(default=False)
    started: bool = Field(default=False)
    minutes: Optional[int] = None

    # Scores
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # Expected goals
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None

    # Difficulty
    home_difficulty: int = Field(default=0)
    away_difficulty: int = Field(default=0)

    # Match stats
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_passes: Optional[int] = None
    away_passes: Optional[int] = None
    home_pass_accuracy: Optional[float] = None
    away_pass_accuracy: Optional[float] = None
    home_offsides: Optional[int] = None
    away_offsides: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    home_saves: Optional[int] = None
    away_saves: Optional[int] = None

    @classmethod
    def from_fpl_api(cls, fixture: dict, team_map: dict) -> "Fixture":
        """Create Fixture from FPL API fixture data."""
        home_team_id = fixture["team_h"]
        away_team_id = fixture["team_a"]

        return cls(
            fpl_id=fixture["id"],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team_name=team_map.get(home_team_id, f"Team {home_team_id}"),
            away_team_name=team_map.get(away_team_id, f"Team {away_team_id}"),
            gameweek=fixture.get("event", 0),
            kickoff_time=fixture.get("kickoff_time"),
            date=fixture.get("kickoff_time", "").split("T")[0] if fixture.get("kickoff_time") else None,
            season=fixture.get("season"),
            finished=fixture.get("finished", False),
            finished_provisional=fixture.get("finished_provisional", False),
            started=fixture.get("started", False),
            minutes=fixture.get("minutes"),
            home_score=fixture.get("team_h_score"),
            away_score=fixture.get("team_a_score"),
            home_xg=fixture.get("team_h_xG"),
            away_xg=fixture.get("team_a_xG"),
            home_difficulty=fixture.get("team_h_difficulty", 0),
            away_difficulty=fixture.get("team_a_difficulty", 0),
            home_shots=fixture.get("team_h_shots"),
            away_shots=fixture.get("team_a_shots"),
            home_shots_on_target=fixture.get("team_h_shots_on_target"),
            away_shots_on_target=fixture.get("team_a_shots_on_target"),
            home_corners=fixture.get("team_h_corners"),
            away_corners=fixture.get("team_a_corners"),
            home_fouls=fixture.get("team_h_fouls"),
            away_fouls=fixture.get("team_a_fouls"),
            home_possession=fixture.get("team_h_possession"),
            away_possession=fixture.get("team_a_possession"),
            home_passes=fixture.get("team_h_passes"),
            away_passes=fixture.get("team_a_passes"),
            home_pass_accuracy=fixture.get("team_h_pass_accuracy"),
            away_pass_accuracy=fixture.get("team_a_pass_accuracy"),
            home_offsides=fixture.get("team_h_offsides"),
            away_offsides=fixture.get("team_a_offsides"),
            home_yellow_cards=fixture.get("team_h_yellow_cards"),
            away_yellow_cards=fixture.get("team_a_yellow_cards"),
            home_red_cards=fixture.get("team_h_red_cards"),
            away_red_cards=fixture.get("team_a_red_cards"),
            home_saves=fixture.get("team_h_saves"),
            away_saves=fixture.get("team_a_saves"),
        )

    def to_csv_row(self) -> dict:
        """Convert to CSV row dictionary."""
        return self.model_dump(exclude_none=True, by_alias=False)
