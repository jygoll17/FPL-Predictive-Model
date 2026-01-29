"""Main CLI entry point."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors import (
    ClubCollector,
    FixtureCollector,
    GameweekCollector,
    PlayerCollector,
)
from src.storage import CSVHandler


async def collect_all_data():
    """Collect all data types."""
    handler = CSVHandler()

    async with PlayerCollector() as collector:
        print("Collecting players...")
        players = await collector.collect()
        handler.save_players(players)
        print(f"Collected {len(players)} players")

    async with ClubCollector() as collector:
        print("Collecting clubs...")
        clubs = await collector.collect()
        handler.save_clubs(clubs)
        print(f"Collected {len(clubs)} clubs")

    async with FixtureCollector() as collector:
        print("Collecting fixtures...")
        fixtures = await collector.collect()
        handler.save_fixtures(fixtures)
        print(f"Collected {len(fixtures)} fixtures")

    async with GameweekCollector() as collector:
        print("Collecting gameweek stats...")
        performances = await collector.collect()
        player_map = handler.get_player_id_map()
        handler.save_gameweek_stats(performances, player_map)
        print(f"Collected {len(performances)} gameweek performances")


if __name__ == "__main__":
    asyncio.run(collect_all_data())
