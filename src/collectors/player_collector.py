"""Player data collector."""

from typing import List

from src.config import FPL_BOOTSTRAP_URL, POSITION_MAP
from src.models.player import Player

from .base import BaseCollector


class PlayerCollector(BaseCollector):
    """Collects player data from FPL API."""

    async def collect(self) -> List[Player]:
        """Collect all players from FPL API."""
        data = await self.fetch_json(FPL_BOOTSTRAP_URL)
        if not data:
            return []

        elements = data.get("elements", [])
        players = []

        for idx, element in enumerate(elements, start=1):
            player = Player.from_fpl_api(element, POSITION_MAP)
            player.player_id = idx
            players.append(player)

        return players
