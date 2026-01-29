"""Fixture data collector."""

from typing import Dict, List

from src.config import FPL_BOOTSTRAP_URL, FPL_FIXTURES_URL
from src.models.fixture import Fixture

from .base import BaseCollector


class FixtureCollector(BaseCollector):
    """Collects fixture data from FPL API."""

    async def collect(self) -> List[Fixture]:
        """Collect all fixtures from FPL API."""
        # Get team names mapping
        bootstrap_data = await self.fetch_json(FPL_BOOTSTRAP_URL)
        if not bootstrap_data:
            return []

        team_map: Dict[int, str] = {}
        for team in bootstrap_data.get("teams", []):
            team_map[team["id"]] = team["name"]

        # Get fixtures
        fixtures_data = await self.fetch_json(FPL_FIXTURES_URL)
        if not fixtures_data:
            return []

        fixtures = []
        for idx, fixture_data in enumerate(fixtures_data, start=1):
            fixture = Fixture.from_fpl_api(fixture_data, team_map)
            fixture.fixture_id = idx
            fixtures.append(fixture)

        return fixtures
