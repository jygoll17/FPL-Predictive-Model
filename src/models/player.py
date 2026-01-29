"""Player data model."""

from typing import Optional
from pydantic import BaseModel, Field


class Player(BaseModel):
    """Player model with all FPL player statistics."""

    # IDs
    player_id: Optional[int] = None  # Our sequential ID
    fpl_id: int = Field(..., description="FPL element ID")
    team_id: int = Field(..., description="Club FPL ID")
    team_code: int = Field(..., description="Club code")

    # Names
    name: str = Field(..., description="Full name")
    first_name: str = Field(..., description="First name")
    second_name: str = Field(..., description="Last name")
    web_name: str = Field(..., description="Display name")

    # Position
    position: str = Field(..., description="GKP/DEF/MID/FWD")
    element_type: int = Field(..., description="Position code (1-4)")

    # Price
    price: float = Field(..., description="Current price in millions")
    cost_change_start: int = Field(default=0, description="Price change from start")
    cost_change_event: int = Field(default=0, description="Price change this GW")

    # Ownership
    selected_by_percent: float = Field(default=0.0, description="Ownership %")
    transfers_in: int = Field(default=0)
    transfers_out: int = Field(default=0)
    transfers_in_event: int = Field(default=0)
    transfers_out_event: int = Field(default=0)

    # Points
    total_points: int = Field(default=0)
    points_per_game: float = Field(default=0.0)
    event_points: int = Field(default=0)
    form: float = Field(default=0.0)
    value_form: float = Field(default=0.0)
    value_season: float = Field(default=0.0)

    # Playing time
    minutes: int = Field(default=0)
    starts: int = Field(default=0)
    chance_of_playing_this_round: Optional[int] = None
    chance_of_playing_next_round: Optional[int] = None

    # News
    news: str = Field(default="")
    news_added: Optional[str] = None

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
    influence_rank: Optional[int] = None
    influence_rank_type: Optional[int] = None
    creativity_rank: Optional[int] = None
    creativity_rank_type: Optional[int] = None
    threat_rank: Optional[int] = None
    threat_rank_type: Optional[int] = None
    ict_index_rank: Optional[int] = None
    ict_index_rank_type: Optional[int] = None

    # Expected stats
    expected_goals: float = Field(default=0.0, alias="xG")
    expected_assists: float = Field(default=0.0, alias="xA")
    expected_goal_involvements: float = Field(default=0.0, alias="xGI")
    expected_goals_conceded: float = Field(default=0.0, alias="xGC")
    expected_goals_per_90: float = Field(default=0.0, alias="xG_per_90")
    expected_assists_per_90: float = Field(default=0.0, alias="xA_per_90")
    expected_goal_involvements_per_90: float = Field(
        default=0.0, alias="xGI_per_90"
    )
    expected_goals_conceded_per_90: float = Field(default=0.0, alias="xGC_per_90")
    goals_conceded_per_90: float = Field(default=0.0)
    saves_per_90: float = Field(default=0.0)
    clean_sheets_per_90: float = Field(default=0.0)
    starts_per_90: float = Field(default=0.0)

    # Status
    status: str = Field(default="")
    in_dreamteam: bool = Field(default=False)
    dreamteam_count: int = Field(default=0)

    class Config:
        populate_by_name = True

    @classmethod
    def from_fpl_api(cls, element: dict, position_map: dict) -> "Player":
        """Create Player from FPL API element."""
        return cls(
            fpl_id=element["id"],
            team_id=element["team"],
            team_code=element["team_code"],
            name=f"{element.get('first_name', '')} {element.get('second_name', '')}".strip(),
            first_name=element.get("first_name", ""),
            second_name=element.get("second_name", ""),
            web_name=element.get("web_name", ""),
            position=position_map.get(element["element_type"], "UNK"),
            element_type=element["element_type"],
            price=element.get("now_cost", 0) / 10.0,
            cost_change_start=element.get("cost_change_start", 0),
            cost_change_event=element.get("cost_change_event", 0),
            selected_by_percent=float(element.get("selected_by_percent", 0)),
            transfers_in=element.get("transfers_in", 0),
            transfers_out=element.get("transfers_out", 0),
            transfers_in_event=element.get("transfers_in_event", 0),
            transfers_out_event=element.get("transfers_out_event", 0),
            total_points=element.get("total_points", 0),
            points_per_game=float(element.get("points_per_game", 0)),
            event_points=element.get("event_points", 0),
            form=float(element.get("form", 0)),
            value_form=float(element.get("value_form", 0)),
            value_season=float(element.get("value_season", 0)),
            minutes=element.get("minutes", 0),
            starts=element.get("starts", 0),
            chance_of_playing_this_round=element.get("chance_of_playing_this_round"),
            chance_of_playing_next_round=element.get("chance_of_playing_next_round"),
            news=element.get("news", ""),
            news_added=element.get("news_added"),
            goals_scored=element.get("goals_scored", 0),
            assists=element.get("assists", 0),
            clean_sheets=element.get("clean_sheets", 0),
            goals_conceded=element.get("goals_conceded", 0),
            own_goals=element.get("own_goals", 0),
            penalties_saved=element.get("penalties_saved", 0),
            penalties_missed=element.get("penalties_missed", 0),
            yellow_cards=element.get("yellow_cards", 0),
            red_cards=element.get("red_cards", 0),
            saves=element.get("saves", 0),
            bonus=element.get("bonus", 0),
            bps=element.get("bps", 0),
            influence=float(element.get("influence", 0)),
            creativity=float(element.get("creativity", 0)),
            threat=float(element.get("threat", 0)),
            ict_index=float(element.get("ict_index", 0)),
            influence_rank=element.get("influence_rank"),
            influence_rank_type=element.get("influence_rank_type"),
            creativity_rank=element.get("creativity_rank"),
            creativity_rank_type=element.get("creativity_rank_type"),
            threat_rank=element.get("threat_rank"),
            threat_rank_type=element.get("threat_rank_type"),
            ict_index_rank=element.get("ict_index_rank"),
            ict_index_rank_type=element.get("ict_index_rank_type"),
            expected_goals=float(element.get("expected_goals", 0)),
            expected_assists=float(element.get("expected_assists", 0)),
            expected_goal_involvements=float(element.get("expected_goal_involvements", 0)),
            expected_goals_conceded=float(element.get("expected_goals_conceded", 0)),
            expected_goals_per_90=float(element.get("expected_goals_per_90", 0)),
            expected_assists_per_90=float(element.get("expected_assists_per_90", 0)),
            expected_goal_involvements_per_90=float(
                element.get("expected_goal_involvements_per_90", 0)
            ),
            expected_goals_conceded_per_90=float(
                element.get("expected_goals_conceded_per_90", 0)
            ),
            goals_conceded_per_90=float(element.get("goals_conceded_per_90", 0)),
            saves_per_90=float(element.get("saves_per_90", 0)),
            clean_sheets_per_90=float(element.get("clean_sheets_per_90", 0)),
            starts_per_90=float(element.get("starts_per_90", 0)),
            status=element.get("status", ""),
            in_dreamteam=element.get("in_dreamteam", False),
            dreamteam_count=element.get("dreamteam_count", 0),
        )

    def to_csv_row(self) -> dict:
        """Convert to CSV row dictionary."""
        return self.model_dump(exclude_none=True, by_alias=False)
