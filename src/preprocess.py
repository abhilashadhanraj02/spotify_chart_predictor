"""
spotify_analysis/src/preprocess.py
─────────────────────────────────────
Cleans the raw dataset and produces train/test splits ready for modelling.

Steps:
  1. Drop irrelevant / high-cardinality text columns
  2. Handle missing values (audio features → median, metadata → mode/fill)
  3. Encode categoricals (genre, key, mode, time_signature)
  4. Normalise continuous features (loudness, tempo)
  5. Add interaction terms
  6. Stratified train / validation / test split (70 / 15 / 15)
  7. Save splits as parquet + feature list as JSON
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PROC_DIR = DATA_DIR / "processed"

# Features that feed the model (after engineering)
NUMERIC_FEATURES = [
    "danceability", "energy", "loudness_norm", "tempo_norm",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "energy_dance", "loud_energy",
    "duration_min", "explicit", "markets_count",
]

CATEGORICAL_FEATURES = [
    "genre", "key", "mode", "time_signature",
]

DROP_COLUMNS = [
    "track_name", "artist_names", "album_name",
    "release_date", "artist_ids", "duration_ms",
    "loudness", "tempo",                         # replaced by normalised versions
    "track_id",                                   # ID not useful as a feature
]


COLUMN_RENAMES = {
    "id":         "track_id",
    "name":       "track_name",
    "artists":    "artist_names",
    "id_artists": "artist_ids",
}

def load_raw(path=None):
    if path is None:
        path = PROC_DIR / "spotify_tracks.csv"
    if not Path(path).exists():
        path = DATA_DIR / "raw" / "checkpoint.parquet"
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Rename Kaggle columns to internal names
    df = df.rename(columns=COLUMN_RENAMES)

    # Add charted label if not present
    if "charted" not in df.columns and "popularity" in df.columns:
        df["charted"] = (df["popularity"] >= 70).astype(int)

    # Add genre placeholder if missing
    if "genre" not in df.columns:
        df["genre"] = "unknown"

    # Derived features if missing
    if "loudness_norm" not in df.columns and "loudness" in df.columns:
        df["loudness_norm"] = ((df["loudness"] - (-60)) / 60).clip(0, 1).round(4)
    if "tempo_norm" not in df.columns and "tempo" in df.columns:
        df["tempo_norm"] = ((df["tempo"] - 40) / 180).clip(0, 1).round(4)
    if "duration_min" not in df.columns and "duration_ms" in df.columns:
        df["duration_min"] = (df["duration_ms"] / 60_000).round(2)
    if "energy_dance" not in df.columns:
        df["energy_dance"] = (df["energy"] * df["danceability"]).round(4)
    if "loud_energy" not in df.columns:
        df["loud_energy"] = (df["loudness_norm"] * df["energy"]).round(4)
    if "markets_count" not in df.columns:
        df["markets_count"] = 0
    if "release_year" not in df.columns and "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audio features: impute with column median (robust to outliers).
    Categorical: impute with mode or 'unknown'.
    """
    af_cols = [
        "danceability", "energy", "loudness_norm", "tempo_norm",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "energy_dance", "loud_energy",
        "key", "mode", "time_signature",
    ]
    for col in af_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ["genre"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    for col in ["release_year"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ["markets_count", "explicit", "duration_min"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode genre.
    key, mode, time_signature are already integers — one-hot encode them.
    Returns (df_encoded, encoder_map).
    """
    encoders = {}

    # Genre → integer label
    if "genre" in df.columns:
        le = LabelEncoder()
        df["genre_enc"] = le.fit_transform(df["genre"].astype(str))
        encoders["genre"] = {cls: int(i) for i, cls in enumerate(le.classes_)}
        df = df.drop(columns=["genre"])

    # key (0–11) → one-hot (drop_first to avoid multicollinearity)
    if "key" in df.columns:
        key_dummies = pd.get_dummies(df["key"].astype(int), prefix="key", drop_first=True)
        df = pd.concat([df.drop(columns=["key"]), key_dummies], axis=1)

    # mode (0/1) → keep as integer
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(int)

    # time_signature → one-hot
    if "time_signature" in df.columns:
        ts_dummies = pd.get_dummies(df["time_signature"].astype(int), prefix="ts", drop_first=True)
        df = pd.concat([df.drop(columns=["time_signature"]), ts_dummies], axis=1)

    return df, encoders


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (everything except target + ID cols)."""
    exclude = {"track_id", "charted", "popularity"}
    return [c for c in df.columns if c not in exclude]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size:  float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split: 70% train, 15% val, 15% test.
    Stratified on 'charted' to preserve class ratio in all splits.
    """
    train_val, test = train_test_split(
        df, test_size=test_size,
        stratify=df["charted"],
        random_state=random_state,
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative,
        stratify=train_val["charted"],
        random_state=random_state,
    )
    log.info(
        "Split: train=%d  val=%d  test=%d  (%.0f/%.0f/%.0f%%)",
        len(train), len(val), len(test),
        len(train)/len(df)*100, len(val)/len(df)*100, len(test)/len(df)*100,
    )
    return train, val, test


def preprocess(raw_path: Path = None, save: bool = True) -> dict:
    """
    Full preprocessing pipeline.

    Returns dict with keys: train, val, test, feature_cols, encoders.
    """
    df = load_raw(raw_path)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    df = drop_columns(df)
    df = handle_missing(df)
    df, encoders = encode_categoricals(df)

    feature_cols = get_feature_columns(df)
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    train, val, test = split_data(df)

    result = {
        "train":        train,
        "val":          val,
        "test":         test,
        "feature_cols": feature_cols,
        "encoders":     encoders,
    }

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        train.to_parquet(PROC_DIR / "train.parquet",  index=False)
        val.to_parquet(  PROC_DIR / "val.parquet",    index=False)
        test.to_parquet( PROC_DIR / "test.parquet",   index=False)

        meta = {
            "feature_cols": feature_cols,
            "encoders":     encoders,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "chart_threshold": 70,
            "charted_rate_train": round(float(train["charted"].mean()), 4),
        }
        (PROC_DIR / "feature_meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Splits + feature_meta.json saved → %s", PROC_DIR)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = preprocess()

    train = result["train"]
    print("\n── Train set sample ──")
    print(train[result["feature_cols"][:6]].head())
    print(f"\nClass balance (train): {train['charted'].value_counts().to_dict()}")
