"""
spotify_analysis/run_pipeline.py
──────────────────────────────────
Single entry point.  Run this to execute the full data collection pipeline:

  python run_pipeline.py [--skip-collect] [--skip-validate] [--no-save]

Flags:
  --skip-collect    load existing checkpoint instead of hitting the API
  --skip-validate   skip the data quality report step
  --no-save         do not write output files (dry run)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from collector   import collect
from validate    import run_validation
from preprocess  import preprocess
from train       import train, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

PROC_DIR = Path(__file__).parent / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Spotify chart-prediction data pipeline")
    parser.add_argument("--skip-collect",   action="store_true", help="Skip API collection, use cached data")
    parser.add_argument("--skip-validate",  action="store_true", help="Skip validation report")
    parser.add_argument("--skip-train",     action="store_true", help="Skip ML training step")
    parser.add_argument("--no-save",        action="store_true", help="Dry run — don't write files")
    args = parser.parse_args()

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Spotify Chart Predictor — Data Pipeline")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Step 1: Collect ───────────────────────────────────────────────────────
    if args.skip_collect:
        log.info("[1/3] Skipping collection — loading cached data")
        from validate import load_data
        df = load_data()
    else:
        log.info("[1/3] Collecting tracks from Spotify API…")
        df = collect(resume=True)

    log.info("      Dataset: %d tracks, %d columns", len(df), len(df.columns))

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    if not args.skip_validate:
        log.info("[2/3] Validating data quality…")
        report = run_validation(df, save_report=not args.no_save)
        if not report.get("quality_pass"):
            log.warning("Quality issues detected — review validation_report.json before modelling")
    else:
        log.info("[2/3] Skipping validation")

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    log.info("[3/3] Preprocessing and splitting…")
    result = preprocess(save=not args.no_save)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        log.info("[4/4] Training ML models…")
        ml_results, _, _, _ = train(use_smote=True, save=not args.no_save)
        print_summary(ml_results)
    else:
        log.info("[4/4] Skipping ML training")

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Pipeline complete ✓")
    log.info("  train : %d rows", len(result["train"]))
    log.info("  val   : %d rows", len(result["val"]))
    log.info("  test  : %d rows", len(result["test"]))
    log.info("  feats : %d features", len(result["feature_cols"]))
    if not args.no_save:
        log.info("  outputs → %s", PROC_DIR)
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
