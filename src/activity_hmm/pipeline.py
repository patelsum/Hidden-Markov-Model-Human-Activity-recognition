"""HMM and baseline pipeline for smartphone human activity recognition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass
class PipelineConfig:
    """Configuration for preprocessing, HMM fitting, and evaluation."""

    subject_column: str = "subject"
    target_column: str = "Activity"
    pca_components: int = 80
    hmm_states: int = 3
    hmm_iterations: int = 100
    window_size: int = 16
    random_state: int = 42


def load_har_data(
    train_path: Path, test_path: Path, target_column: str = "Activity"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the benchmark train/test CSV files."""

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if target_column not in train.columns or target_column not in test.columns:
        raise ValueError("Both CSV files must contain the Activity target column.")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train and test CSV files must have identical column order.")
    if train.empty or test.empty:
        raise ValueError("Train and test CSV files must not be empty.")
    return train, test


def _sequence_bounds(
    frame: pd.DataFrame, subject_column: str, target_column: Optional[str]
) -> Iterator[Tuple[int, int]]:
    """Yield contiguous subject or subject/activity sequence boundaries."""

    if frame.empty:
        return

    if subject_column in frame.columns:
        subjects = frame[subject_column].to_numpy()
    else:
        subjects = np.zeros(len(frame), dtype=int)

    targets = None
    if target_column and target_column in frame.columns:
        targets = frame[target_column].to_numpy()

    start = 0
    for index in range(1, len(frame)):
        subject_changed = subjects[index] != subjects[index - 1]
        target_changed = targets is not None and targets[index] != targets[index - 1]
        if subject_changed or target_changed:
            yield start, index
            start = index
    yield start, len(frame)


def _subject_windows(
    frame: pd.DataFrame, subject_column: str, window_size: int
) -> Iterator[Tuple[int, int]]:
    """Yield fixed-size prediction windows without crossing subject boundaries."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    if frame.empty:
        return

    if subject_column in frame.columns:
        subjects = frame[subject_column].to_numpy()
    else:
        subjects = np.zeros(len(frame), dtype=int)

    start = 0
    for index in range(1, len(frame) + 1):
        at_subject_end = index == len(frame) or subjects[index] != subjects[index - 1]
        if at_subject_end:
            subject_end = index
            for window_start in range(start, subject_end, window_size):
                yield window_start, min(window_start + window_size, subject_end)
            start = index


def _json_safe_report(report: Dict[str, object]) -> Dict[str, object]:
    """Convert sklearn report values to JSON-safe Python types."""

    result: Dict[str, object] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            result[key] = {metric: float(metric_value) for metric, metric_value in value.items()}
        else:
            result[key] = float(value)
    return result


class ActivityRecognitionPipeline:
    """Train class-conditional HMMs and a Logistic Regression baseline."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.feature_columns: List[str] = []
        self.labels: List[str] = []
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.hmm_models: Dict[str, GaussianHMM] = {}
        self.baseline: Optional[LogisticRegression] = None

    def _select_features(self, frame: pd.DataFrame) -> List[str]:
        excluded = {self.config.target_column, self.config.subject_column}
        return [column for column in frame.columns if column not in excluded]

    def _check_features(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError("Input is missing feature columns: " + ", ".join(missing[:5]))

    def _fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.feature_columns].to_numpy(dtype=float)
        scaled = self.scaler.fit_transform(raw)
        component_count = min(self.config.pca_components, scaled.shape[0], scaled.shape[1])
        if component_count < 1:
            raise ValueError("The training data must contain at least one feature.")
        self.pca = PCA(n_components=component_count, random_state=self.config.random_state)
        return self.pca.fit_transform(scaled)

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("The pipeline must be fitted before prediction.")
        self._check_features(frame)
        raw = frame[self.feature_columns].to_numpy(dtype=float)
        return self.pca.transform(self.scaler.transform(raw))

    def fit(self, train: pd.DataFrame) -> "ActivityRecognitionPipeline":
        """Fit preprocessing, one HMM per label, and the baseline classifier."""

        target = self.config.target_column
        if target not in train.columns:
            raise ValueError("Training data must contain the Activity target column.")
        self.feature_columns = self._select_features(train)
        if not self.feature_columns:
            raise ValueError("No model features remain after excluding identifier and target.")

        transformed = self._fit_transform(train)
        target_values = train[target].to_numpy()
        self.labels = sorted(str(label) for label in pd.unique(target_values))

        for label in self.labels:
            sequences: List[np.ndarray] = []
            lengths: List[int] = []
            for start, end in _sequence_bounds(train, self.config.subject_column, target):
                if str(target_values[start]) == label:
                    sequences.append(transformed[start:end])
                    lengths.append(end - start)
            if not sequences:
                raise ValueError("No training sequence found for activity: " + label)

            observations = np.vstack(sequences)
            state_count = max(1, min(self.config.hmm_states, min(lengths)))
            model = GaussianHMM(
                n_components=state_count,
                covariance_type="diag",
                n_iter=self.config.hmm_iterations,
                min_covar=1e-3,
                random_state=self.config.random_state,
            )
            model.fit(observations, lengths=lengths)
            self.hmm_models[label] = model

        self.baseline = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            random_state=self.config.random_state,
        )
        self.baseline.fit(transformed, target_values.astype(str))
        return self

    def predict_hmm(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict one activity label per row using subject-safe windows."""

        if not self.hmm_models:
            raise RuntimeError("The pipeline must be fitted before prediction.")
        transformed = self._transform(frame)
        predictions = np.empty(len(frame), dtype=object)
        for start, end in _subject_windows(
            frame, self.config.subject_column, self.config.window_size
        ):
            window = transformed[start:end]
            scores = {label: model.score(window) for label, model in self.hmm_models.items()}
            prediction = max(scores, key=scores.get)
            predictions[start:end] = prediction
        return predictions

    def predict_baseline(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict one activity label per row using Logistic Regression."""

        if self.baseline is None:
            raise RuntimeError("The pipeline must be fitted before prediction.")
        return self.baseline.predict(self._transform(frame))

    def _metrics(self, actual: Sequence[object], predicted: Sequence[object]) -> Dict[str, object]:
        actual_text = np.asarray(actual).astype(str)
        predicted_text = np.asarray(predicted).astype(str)
        report = classification_report(
            actual_text,
            predicted_text,
            labels=self.labels,
            target_names=self.labels,
            output_dict=True,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(actual_text, predicted_text)),
            "f1_macro": float(f1_score(actual_text, predicted_text, average="macro")),
            "f1_weighted": float(f1_score(actual_text, predicted_text, average="weighted")),
            "classification_report": _json_safe_report(report),
        }

    def evaluate(self, test: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        """Evaluate both models and return JSON-serializable metrics."""

        target = self.config.target_column
        if target not in test.columns:
            raise ValueError("Test data must contain the Activity target column.")
        actual = test[target].to_numpy()
        return {
            "hmm": self._metrics(actual, self.predict_hmm(test)),
            "logistic_regression": self._metrics(actual, self.predict_baseline(test)),
        }
