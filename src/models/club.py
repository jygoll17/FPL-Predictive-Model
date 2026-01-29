"""Club data model."""

from typing import Optional
from pydantic import BaseModel, Field


class Club(BaseModel):
    """Club/Team model with FPL team data."""

    # IDs
    club_id: Optional[int] = None  # Our sequential ID
    fpl_id: int = Field(..., description="FPL team ID")
    code: int = Field(..., description="Team code")
    pulse_id: Optional[int] = None

    # Names
    name: str = Field(..., description="Full name")
    short_name: str = Field(..., description="Short name")

    # Strength ratings
    strength: int = Field(default=0)
    strength_overall_home: int = Field(default=0)
    strength_overall_away: int = Field(default=0)
    strength_attack_home: int = Field(default=0)
    strength_attack_away: int = Field(default=0)
    strength_defence_home: int = Field(default=0)
    strength_defence_away: int = Field(default=0)

    # Season stats
    played: int = Field(default=0)
    wins: int = Field(default=0)
    draws: int = Field(default=0)
    losses: int = Field(default=0)
    goals_scored: int = Field(default=0)
    goals_conceded: int = Field(default=0)
    goal_difference: int = Field(default=0)
    points: int = Field(default=0)
    position: int = Field(default=0)

    # Advanced stats
    xg: float = Field(default=0.0)
    xga: float = Field(default=0.0)
    npxg: float = Field(default=0.0)
    shots: int = Field(default=0)
    shots_on_target: int = Field(default=0)
    big_chances_created: int = Field(default=0)
    big_chances_missed: int = Field(default=0)
    possession: float = Field(default=0.0)
    passes: int = Field(default=0)
    pass_accuracy: float = Field(default=0.0)
    tackles: int = Field(default=0)
    interceptions: int = Field(default=0)
    clearances: int = Field(default=0)
    clean_sheets: int = Field(default=0)
    saves: int = Field(default=0)
    fouls: int = Field(default=0)
    yellow_cards: int = Field(default=0)
    red_cards: int = Field(default=0)
    corners: int = Field(default=0)
    offsides: int = Field(default=0)

    # Status
    unavailable: bool = Field(default=False)

    @classmethod
    def from_fpl_api(cls, team: dict) -> "Club":
        """Create Club from FPL API team data."""
        return cls(
            fpl_id=team["id"],
            code=team["code"],
            name=team["name"],
            short_name=team["short_name"],
            pulse_id=team.get("pulse_id"),
            strength=team.get("strength", 0),
            strength_overall_home=team.get("strength_overall_home", 0),
            strength_overall_away=team.get("strength_overall_away", 0),
            strength_attack_home=team.get("strength_attack_home", 0),
            strength_attack_away=team.get("strength_attack_away", 0),
            strength_defence_home=team.get("strength_defence_home", 0),
            strength_defence_away=team.get("strength_defence_away", 0),
            played=team.get("played", 0),
            wins=team.get("win", 0),
            draws=team.get("draw", 0),
            losses=team.get("loss", 0),
            goals_scored=team.get("goals_scored", 0),
            goals_conceded=team.get("goals_conceded", 0),
            goal_difference=team.get("goal_difference", 0),
            points=team.get("points", 0),
            position=team.get("position", 0),
            unavailable=team.get("unavailable", False),
        )

    def to_csv_row(self) -> dict:
        """Convert to CSV row dictionary."""
        return self.model_dump(exclude_none=True, by_alias=False)
