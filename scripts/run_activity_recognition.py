"""Run the reusable human activity recognition pipeline."""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from activity_hmm.pipeline import ActivityRecognitionPipeline, PipelineConfig, load_har_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/metrics.json"), help="Metrics JSON path"
    )
    parser.add_argument("--pca-components", type=int, default=80)
    parser.add_argument("--hmm-states", type=int, default=3)
    parser.add_argument("--hmm-iterations", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, test = load_har_data(args.train, args.test)
    config = PipelineConfig(
        pca_components=args.pca_components,
        hmm_states=args.hmm_states,
        hmm_iterations=args.hmm_iterations,
        window_size=args.window_size,
    )
    pipeline = ActivityRecognitionPipeline(config).fit(train)
    metrics = pipeline.evaluate(test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Activities:", ", ".join(pipeline.labels))
    print("HMM accuracy:", round(metrics["hmm"]["accuracy"], 4))
    print("Logistic Regression accuracy:", round(metrics["logistic_regression"]["accuracy"], 4))
    print("Metrics written to:", args.output)


if __name__ == "__main__":
    main()
