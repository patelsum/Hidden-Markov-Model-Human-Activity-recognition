import numpy as np
import pandas as pd

from activity_hmm.pipeline import ActivityRecognitionPipeline, PipelineConfig


def _synthetic_activity_data(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = {"REST": (-2.0, 0.0), "MOVE": (2.0, 0.0), "CLIMB": (0.0, 2.0)}

    def build(subject_start: int, subjects: int, rows_per_subject: int) -> pd.DataFrame:
        records = []
        for subject in range(subject_start, subject_start + subjects):
            label = list(centers)[(subject - subject_start) % len(centers)]
            center_x, center_y = centers[label]
            for _ in range(rows_per_subject):
                x, y = rng.normal((center_x, center_y), 0.15)
                records.append({"subject": subject, "feature_x": x, "feature_y": y, "Activity": label})
        return pd.DataFrame(records)

    return build(1, 9, 10), build(100, 6, 8)


def test_pipeline_fits_and_returns_metrics():
    train, test = _synthetic_activity_data()
    pipeline = ActivityRecognitionPipeline(
        PipelineConfig(pca_components=2, hmm_states=2, hmm_iterations=20, window_size=4)
    ).fit(train)

    metrics = pipeline.evaluate(test)

    assert set(metrics) == {"hmm", "logistic_regression"}
    assert set(metrics["hmm"]) == {"accuracy", "f1_macro", "f1_weighted", "classification_report"}
    assert 0.0 <= metrics["hmm"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["logistic_regression"]["accuracy"] <= 1.0
