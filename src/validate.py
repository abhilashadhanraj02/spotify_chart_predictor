"""
spotify_analysis/src/validate.py
──────────────────────────────────
Validates the collected dataset and generates a concise data quality report.
Run after collector.py to confirm the dataset is clean before modelling.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
]

# Kaggle dataset column mapping → our internal names
COLUMN_RENAMES = {
    "id":         "track_id",
    "name":       "track_name",
    "artists":    "artist_names",
    "id_artists": "artist_ids",
}

EXPECTED_RANGES = {
    "danceability":     (0.0, 1.0),
    "energy":           (0.0, 1.0),
    "key":              (0, 11),
    "loudness":         (-60.0, 0.0),
    "mode":             (0, 1),
    "speechiness":      (0.0, 1.0),
    "acousticness":     (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness":         (0.0, 1.0),
    "valence":          (0.0, 1.0),
    "tempo":            (30.0, 250.0),
    "time_signature":   (1, 7),
    "popularity":       (0, 100),
    "duration_ms":      (10_000, 3_600_000),
}


def load_data(path=None):
    if path is None:
        path = PROC_DIR / "spotify_tracks.csv"
    if not Path(path).exists():
        path = RAW_DIR / "checkpoint.parquet"
    if not Path(path).exists():
        raise FileNotFoundError("No dataset found. Run collector.py first.")
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # Rename Kaggle columns to internal names
    df = df.rename(columns=COLUMN_RENAMES)

    # Add charted label if not present
    if "charted" not in df.columns and "popularity" in df.columns:
        df["charted"] = (df["popularity"] >= 70).astype(int)

    # Add placeholder genre if missing
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


def check_missing(df: pd.DataFrame) -> dict:
    """Return columns with missing value counts and percentages."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return {
        col: {"count": int(cnt), "pct": round(cnt / len(df) * 100, 2)}
        for col, cnt in missing.items()
    }


def check_ranges(df: pd.DataFrame) -> dict:
    """Return columns where values fall outside expected ranges."""
    violations = {}
    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out = ((series < lo) | (series > hi)).sum()
        if out > 0:
            violations[col] = {
                "out_of_range": int(out),
                "pct": round(out / len(df) * 100, 2),
                "expected": [lo, hi],
                "actual_min": round(float(series.min()), 4),
                "actual_max": round(float(series.max()), 4),
            }
    return violations


def check_duplicates(df: pd.DataFrame) -> dict:
    dups = df.duplicated(subset="track_id").sum()
    return {"duplicate_track_ids": int(dups)}


def feature_stats(df: pd.DataFrame) -> dict:
    """Descriptive stats for all audio features."""
    stats = {}
    for col in AUDIO_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].describe()
        stats[col] = {
            "mean":   round(float(s["mean"]), 4),
            "std":    round(float(s["std"]),  4),
            "min":    round(float(s["min"]),  4),
            "25%":    round(float(s["25%"]),  4),
            "median": round(float(s["50%"]),  4),
            "75%":    round(float(s["75%"]),  4),
            "max":    round(float(s["max"]),  4),
        }
    return stats


def class_balance(df: pd.DataFrame) -> dict:
    counts = df["charted"].value_counts().to_dict()
    total  = len(df)
    return {
        "charted":     {"count": int(counts.get(1, 0)), "pct": round(counts.get(1, 0) / total * 100, 2)},
        "non_charted": {"count": int(counts.get(0, 0)), "pct": round(counts.get(0, 0) / total * 100, 2)},
        "imbalance_ratio": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 2),
    }


def genre_breakdown(df: pd.DataFrame) -> dict:
    if "genre" not in df.columns:
        return {}
    gb = df.groupby("genre").agg(
        count=("track_id", "count"),
        charted_rate=("charted", "mean"),
        avg_popularity=("popularity", "mean"),
    ).round(4)
    return gb.to_dict(orient="index")


def correlation_with_target(df: pd.DataFrame) -> dict:
    corrs = {}
    for col in AUDIO_FEATURES:
        if col in df.columns:
            c = df[col].corr(df["charted"])
            if not np.isnan(c):
                corrs[col] = round(float(c), 4)
    return dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))


def run_validation(df: pd.DataFrame = None, save_report: bool = True) -> dict:
    if df is None:
        df = load_data()

    log.info("Running validation on %d rows…", len(df))

    report = {
        "dataset_overview": {
            "total_tracks":  len(df),
            "total_columns": len(df.columns),
            "columns":       list(df.columns),
        },
        "missing_values":        check_missing(df),
        "out_of_range":          check_ranges(df),
        "duplicates":            check_duplicates(df),
        "class_balance":         class_balance(df),
        "genre_breakdown":       genre_breakdown(df),
        "feature_statistics":    feature_stats(df),
        "correlation_with_chart": correlation_with_target(df),
    }

    # ── Summary pass/fail ─────────────────────────────────────────────────────
    issues = []
    if report["duplicates"]["duplicate_track_ids"] > 0:
        issues.append(f"{report['duplicates']['duplicate_track_ids']} duplicate track IDs")
    for col, v in report["missing_values"].items():
        if v["pct"] > 5:
            issues.append(f"High missingness in {col}: {v['pct']}%")
    for col, v in report["out_of_range"].items():
        if v["pct"] > 1:
            issues.append(f"Out-of-range values in {col}: {v['pct']}%")
    balance = report["class_balance"]["imbalance_ratio"]
    if balance > 10:
        issues.append(f"Severe class imbalance — ratio {balance:.1f}:1  (consider SMOTE)")

    report["quality_issues"] = issues
    report["quality_pass"]   = len(issues) == 0

    if save_report:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / "validation_report.json"
        out.write_text(json.dumps(report, indent=2))
        log.info("Validation report saved → %s", out)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  DATA QUALITY REPORT")
    print("═" * 60)
    print(f"  Total tracks      : {len(df):,}")
    print(f"  Total columns     : {len(df.columns)}")
    print(f"  Charted           : {report['class_balance']['charted']['count']:,}  "
          f"({report['class_balance']['charted']['pct']}%)")
    print(f"  Non-charted       : {report['class_balance']['non_charted']['count']:,}  "
          f"({report['class_balance']['non_charted']['pct']}%)")
    print(f"  Imbalance ratio   : {balance:.1f}:1")
    print(f"  Missing values    : {len(report['missing_values'])} columns affected")
    print(f"  Duplicates        : {report['duplicates']['duplicate_track_ids']}")
    print()
    print("  Top correlations with 'charted' label:")
    for feat, corr in list(report["correlation_with_chart"].items())[:6]:
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"    {feat:<20} {sign}{abs(corr):.4f}  {bar}")
    print()
    if issues:
        print("  ⚠️  Quality issues:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print("  ✓  All quality checks passed")
    print("═" * 60 + "\n")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_validation()
