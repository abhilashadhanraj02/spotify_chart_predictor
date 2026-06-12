"""
spotify_analysis/src/predictor.py
───────────────────────────────────
Load the saved best model and predict chart probability for new tracks.

Usage:
    from predictor import Predictor
    p = Predictor()
    prob = p.predict_one(energy=0.8, danceability=0.7, valence=0.6, ...)
    print(f"Chart probability: {prob:.1%}")
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"


class Predictor:
    def __init__(self):
        results      = json.loads((MODELS_DIR / "results.json").read_text())
        best_name    = results["best_model"]
        self.feature_cols = results["feature_cols"]
        self.model   = joblib.load(MODELS_DIR / f"{best_name}.joblib")
        self.imputer = joblib.load(MODELS_DIR / "imputer.joblib")
        log.info("Loaded %s model with %d features", best_name, len(self.feature_cols))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict chart probability for a DataFrame of tracks."""
        X = df.reindex(columns=self.feature_cols, fill_value=0)
        X = self.imputer.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict_one(self, **kwargs) -> float:
        """
        Predict chart probability for a single track.
        Pass audio features as keyword arguments.

        Example:
            p.predict_one(energy=0.8, danceability=0.7, loudness_norm=0.7,
                          valence=0.6, tempo_norm=0.55, acousticness=0.1,
                          speechiness=0.05, liveness=0.12)
        """
        # Build a single-row DataFrame with all expected features as 0
        row = {col: 0.0 for col in self.feature_cols}
        row.update(kwargs)

        # Auto-compute interaction features if base features supplied
        if "energy" in kwargs and "danceability" in kwargs:
            row["energy_dance"] = kwargs["energy"] * kwargs["danceability"]
        if "loudness_norm" in kwargs and "energy" in kwargs:
            row["loud_energy"] = kwargs["loudness_norm"] * kwargs["energy"]

        df = pd.DataFrame([row])
        prob = self.predict_proba(df)[0]
        return float(prob)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = Predictor()

    # Example: high-energy pop track
    pop_hit = p.predict_one(
        energy=0.82, danceability=0.75, loudness_norm=0.74,
        valence=0.65, tempo_norm=0.58, acousticness=0.08,
        speechiness=0.06, liveness=0.12, explicit=0,
    )
    print(f"Pop hit chart probability:     {pop_hit:.1%}")

    # Example: quiet acoustic track
    acoustic = p.predict_one(
        energy=0.28, danceability=0.38, loudness_norm=0.40,
        valence=0.45, tempo_norm=0.35, acousticness=0.88,
        speechiness=0.04, liveness=0.10, explicit=0,
    )
    print(f"Acoustic track chart probability: {acoustic:.1%}")
