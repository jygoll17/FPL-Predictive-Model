"""Club data collector."""

from typing import List

from src.config import FPL_BOOTSTRAP_URL
from src.models.club import Club

from .base import BaseCollector


class ClubCollector(BaseCollector):
    """Collects club/team data from FPL API."""

    async def collect(self) -> List[Club]:
        """Collect all clubs from FPL API."""
        data = await self.fetch_json(FPL_BOOTSTRAP_URL)
        if not data:
            return []

        teams = data.get("teams", [])
        clubs = []

        for idx, team in enumerate(teams, start=1):
            club = Club.from_fpl_api(team)
            club.club_id = idx
            clubs.append(club)

        return clubs
