#!/usr/bin/env python3
"""Data collection script."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.collectors import (
    ClubCollector,
    FixtureCollector,
    GameweekCollector,
    PlayerCollector,
)
from src.storage import CSVHandler


async def collect_players():
    """Collect player data."""
    handler = CSVHandler()
    async with PlayerCollector() as collector:
        print("Collecting players...")
        players = await collector.collect()
        handler.save_players(players)
        print(f"✓ Collected {len(players)} players")


async def collect_clubs():
    """Collect club data."""
    handler = CSVHandler()
    async with ClubCollector() as collector:
        print("Collecting clubs...")
        clubs = await collector.collect()
        handler.save_clubs(clubs)
        print(f"✓ Collected {len(clubs)} clubs")


async def collect_fixtures():
    """Collect fixture data."""
    handler = CSVHandler()
    async with FixtureCollector() as collector:
        print("Collecting fixtures...")
        fixtures = await collector.collect()
        handler.save_fixtures(fixtures)
        print(f"✓ Collected {len(fixtures)} fixtures")


async def collect_gameweeks():
    """Collect gameweek stats."""
    handler = CSVHandler()
    async with GameweekCollector() as collector:
        print("Collecting gameweek stats...")
        performances = await collector.collect()
        player_map = handler.get_player_id_map()
        handler.save_gameweek_stats(performances, player_map)
        print(f"✓ Collected {len(performances)} gameweek performances")


async def show_status():
    """Show data collection status."""
    handler = CSVHandler()
    from src.config import (
        CLUBS_CSV,
        FIXTURE_HISTORY_CSV,
        METADATA_JSON,
        PLAYER_GAMEWEEK_CSV,
        PLAYERS_CSV,
    )

    import json
    from datetime import datetime

    print("\nData Collection Status:")
    print("=" * 50)

    files = {
        "Players": PLAYERS_CSV,
        "Clubs": CLUBS_CSV,
        "Fixtures": FIXTURE_HISTORY_CSV,
        "Gameweek Stats": PLAYER_GAMEWEEK_CSV,
    }

    for name, path in files.items():
        if path.exists():
            import pandas as pd
            df = pd.read_csv(path)
            print(f"{name:20} {len(df):6} rows")
        else:
            print(f"{name:20} {'Not collected':>15}")

    if METADATA_JSON.exists():
        with open(METADATA_JSON, "r") as f:
            metadata = json.load(f)
        print("\nLast Updated:")
        for key, value in metadata.items():
            print(f"  {key:20} {value}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect FPL data")
    parser.add_argument("--players", action="store_true", help="Collect players only")
    parser.add_argument("--clubs", action="store_true", help="Collect clubs only")
    parser.add_argument("--fixtures", action="store_true", help="Collect fixtures only")
    parser.add_argument("--gameweeks", action="store_true", help="Collect gameweek stats only")
    parser.add_argument("--status", action="store_true", help="Show data status")

    args = parser.parse_args()

    if args.status:
        await show_status()
        return

    if args.players:
        await collect_players()
    elif args.clubs:
        await collect_clubs()
    elif args.fixtures:
        await collect_fixtures()
    elif args.gameweeks:
        await collect_gameweeks()
    else:
        # Collect all
        await collect_players()
        await collect_clubs()
        await collect_fixtures()
        await collect_gameweeks()
        print("\n✓ All data collected successfully!")


if __name__ == "__main__":
    asyncio.run(main())
