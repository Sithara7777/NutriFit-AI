"""Cross-check the BMI classification against the UCI obesity dataset.

    python ml/scripts/validate_bmi.py

Optional. If the dataset is not present the script explains how to get it and
exits 0 — nothing in the build depends on it.

Writes ml/reports/bmi_validation.txt and bmi_confusion_matrix.csv for the
Testing and Evaluation chapter.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrifit import config, validation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

DOWNLOAD_HELP = """
The obesity dataset was not found in data/raw/.

This cross-check is OPTIONAL — the system does not depend on it. To run it:

  1. Download from either:
       https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
       https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
  2. Put the CSV in data/raw/ (any filename containing "obesity" is found
     automatically).
  3. Re-run this script.

If you choose NOT to run it, see docs/BMI_VALIDATION.md for the written
justification to use in your report instead.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BMI classification logic.")
    parser.add_argument("--path", type=Path, default=None, help="Explicit path to the CSV.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the dataset is missing (for CI).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = validation.load_obesity_dataset(args.path)
    if df is None:
        print(DOWNLOAD_HELP)
        return 1 if args.strict else 0

    result = validation.cross_check_bmi_classification(df)
    report = validation.format_report(result)
    print("\n" + report + "\n")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "bmi_validation.txt").write_text(report, encoding="utf-8")
    result["confusion_matrix"].to_csv(config.REPORTS_DIR / "bmi_confusion_matrix.csv")
    result["disagreements"].to_csv(
        config.REPORTS_DIR / "bmi_disagreements.csv", index=False
    )

    print(f"Wrote {config.REPORTS_DIR / 'bmi_validation.txt'}")
    print(f"Wrote {config.REPORTS_DIR / 'bmi_confusion_matrix.csv'}")
    print(f"Wrote {config.REPORTS_DIR / 'bmi_disagreements.csv'}")

    if result["n_disagree"]:
        print("\nClosest disagreements (smallest distance to a WHO boundary):")
        print(result["disagreements"].head(10).to_string(index=False))

    # Agreement below this suggests a genuine logic error rather than rounding.
    threshold = 0.95
    if result["agreement_rate"] < threshold:
        print(
            f"\nWARNING: agreement {result['agreement_rate']:.2%} is below "
            f"{threshold:.0%}. Investigate before reporting."
        )
        return 1

    print(f"\nPASS — {result['agreement_rate']:.2%} agreement with the independent labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
