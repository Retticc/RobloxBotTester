#!/usr/bin/env python3
"""Visual Trend Analyzer

A simplified module that loads trending icon and thumbnail data from the
PostgreSQL database and writes a short summary report.  The original file
was truncated in the repository, so this version provides a minimal but
functional implementation.
"""

import os
import json
from datetime import datetime
from typing import Dict

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()


class VisualTrendAnalyzer:
    """Analyze trending visual assets used by Roblox games."""

    def __init__(self, output_dir: str = "visual_trends") -> None:
        self.db_url = os.getenv("DATABASE_URL")
        self.output_dir = output_dir
        self.ensure_output_directory()

    def get_conn(self) -> psycopg2.extensions.connection:
        """Return a new database connection."""
        if not self.db_url:
            raise ValueError("DATABASE_URL not configured")
        return psycopg2.connect(self.db_url, sslmode="require")

    def ensure_output_directory(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)

    def analyze_trending_visuals(self, days_back: int = 30) -> pd.DataFrame:
        """Return a dataframe of the top visual assets over the last `days_back` days."""
        interval = f"{days_back} days"
        query = """
            SELECT game_id, game_name, asset_type, performance_score,
                   AVG(playing) as avg_playing
            FROM snapshots
            WHERE snapshot_time >= NOW() - INTERVAL %s
            GROUP BY game_id, game_name, asset_type, performance_score
            ORDER BY performance_score DESC
            LIMIT 50
        """
        with self.get_conn() as conn:
            df = pd.read_sql(query, conn, params=(interval,))
        return df

    def save_trending_assets_to_database(self, df: pd.DataFrame) -> None:
        """Placeholder for saving trending asset information back to the database."""
        if df.empty:
            return
        # Real implementation would insert rows into a table.
        pass

    def generate_trending_report(self) -> Dict[str, object]:
        """Generate trending report and save summary JSON."""
        df = self.analyze_trending_visuals()
        report = {
            "generated": datetime.utcnow().isoformat(),
            "records": len(df),
        }
        report_path = os.path.join(self.output_dir, "reports", "trend_summary.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    def print_trending_summary(self) -> None:
        """Print a simple summary of today's trending visual styles."""
        query = """
            SELECT visual_style, COUNT(*) AS games,
                   AVG(performance_score) AS avg_score
            FROM trending_visual_assets
            WHERE analysis_date::date = CURRENT_DATE
            GROUP BY visual_style
            ORDER BY avg_score DESC
            LIMIT 5
        """
        with self.get_conn() as conn:
            df = pd.read_sql(query, conn)
        for _, row in df.iterrows():
            print(f"{row['visual_style']}: {row['avg_score']:.2f} ({row['games']} games)")


def main() -> None:
    analyzer = VisualTrendAnalyzer()
    analyzer.generate_trending_report()
    analyzer.print_trending_summary()


if __name__ == "__main__":
    main()
