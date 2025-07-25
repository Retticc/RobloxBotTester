#!/usr/bin/env python3
"""Continuous learning predictor for Roblox games.

This module provides a simplified yet functional version of the
``ContinuousGamePredictor`` class.  The original file was truncated in
previous commits so this version rebuilds the class with the essential
methods required by ``complete_integration.py``.  It can create a
training set from the database, train a basic model and make
predictions for individual games.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

@dataclass
class TrainingResult:
    accuracy: float
    total_games: int
    retrained: bool


class ContinuousGamePredictor:
    """Minimal continuous predictor used by ``complete_integration``."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names = ["avg_playing", "snapshot_count"]
        self.last_training: Optional[datetime] = None
        self.retrain_threshold = 10
        self.min_training_games = 30

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def get_conn(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(self.db_url, sslmode="require")

    def ensure_tracking_tables(self) -> None:
        """Create a minimal metadata table used for bookkeeping."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ml_training_history (
                        id SERIAL PRIMARY KEY,
                        training_date TIMESTAMP DEFAULT NOW(),
                        total_games INTEGER,
                        accuracy FLOAT
                    );
                    """
                )
            conn.commit()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def should_retrain(self) -> bool:
        """Return ``True`` if new games have been added since the last train."""
        if not self.last_training:
            return True
        try:
            with self.get_conn() as conn:
                query = """
                    SELECT COUNT(DISTINCT s.game_id) AS new_games
                    FROM snapshots s
                    WHERE s.snapshot_time > %s
                """
                result = pd.read_sql(query, conn, params=(self.last_training,))
                new_games = int(result.iloc[0]["new_games"]) if len(result) else 0
                return new_games >= self.retrain_threshold
        except Exception:
            return True

    def create_training_dataset(self, min_snapshots: int = 5) -> pd.DataFrame:
        """Return a simple training dataframe."""
        with self.get_conn() as conn:
            query = """
                SELECT g.id, g.name,
                       AVG(s.playing) AS avg_playing,
                       COUNT(s.game_id) AS snapshot_count
                FROM games g
                JOIN snapshots s ON g.id = s.game_id
                GROUP BY g.id, g.name
                HAVING COUNT(s.game_id) >= %s
            """
            df = pd.read_sql(query, conn, params=(min_snapshots,))
        df["success"] = df["avg_playing"] >= 100
        return df

    def train_models(self, df: pd.DataFrame) -> float:
        """Train a simple RandomForest model."""
        X = df[self.feature_names]
        y = df["success"].astype(int)

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        self.model = model
        self.last_training = datetime.utcnow()
        return accuracy

    def continuous_train(self) -> TrainingResult:
        """Entry point used by ``complete_integration``."""
        self.ensure_tracking_tables()
        retrain = self.should_retrain()
        df = self.create_training_dataset()
        total = len(df)
        accuracy = 0.0
        if retrain and total >= self.min_training_games:
            accuracy = self.train_models(df)
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO ml_training_history (total_games, accuracy) VALUES (%s, %s)",
                        (total, accuracy),
                    )
                conn.commit()
        return TrainingResult(accuracy=accuracy, total_games=total, retrained=retrain)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_game_success(self, game_id: int) -> Optional[Dict[str, Any]]:
        """Predict the success probability for ``game_id``."""
        if not self.model:
            return None
        with self.get_conn() as conn:
            query = """
                SELECT g.id, g.name,
                       AVG(s.playing) AS avg_playing,
                       COUNT(s.game_id) AS snapshot_count
                FROM games g
                JOIN snapshots s ON g.id = s.game_id
                WHERE g.id = %s
                GROUP BY g.id, g.name
            """
            result = pd.read_sql(query, conn, params=(game_id,))
            if result.empty:
                return None
            row = result.iloc[0]
        features = np.array([[row["avg_playing"], row["snapshot_count"]]])
        features = self.scaler.transform(features)
        prob = self.model.predict_proba(features)[0, 1]
        pred = bool(self.model.predict(features)[0])
        return {
            "game_id": int(row["id"]),
            "game_name": row["name"],
            "success": pred,
            "probability": float(prob),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_model(self, path: str = "continuous_game_model.pkl") -> None:
        if not self.model:
            return
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "last_training": self.last_training,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def load_model(self, path: str = "continuous_game_model.pkl") -> bool:
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data.get("feature_names", self.feature_names)
            self.last_training = data.get("last_training")
            return True
        except FileNotFoundError:
            return False


def main() -> None:
    predictor = ContinuousGamePredictor()
    result = predictor.continuous_train()
    print(f"trained: {result.retrained}, games: {result.total_games}, accuracy: {result.accuracy:.2%}")
    predictor.save_model()


if __name__ == "__main__":
    main()
