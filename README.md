# Human Activity Recognition with Hidden Markov Models

This repository classifies six human activities from smartphone sensor features and compares a class-conditional Hidden Markov Model (HMM) with a scikit-learn Logistic Regression baseline.

The project is an educational benchmark based on the UCI Human Activity Recognition Using Smartphones dataset. It is designed to show how feature preprocessing, temporal windows, probabilistic sequence scoring, and a conventional classifier fit together.

## Activities

- `LAYING`
- `SITTING`
- `STANDING`
- `WALKING`
- `WALKING_DOWNSTAIRS`
- `WALKING_UPSTAIRS`

## Dataset

Download the dataset from the Kaggle mirror:

https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones

Place the two files below in a local `Data` directory. The raw CSV files are intentionally excluded from Git.

```text
Data/train.csv
Data/test.csv
```

The saved outputs from the original notebook show 7,352 training rows, 2,947 test rows, six activity classes, and 562 input columns before preprocessing. The `Activity` column is the target and `subject` is treated as an identifier rather than a predictive feature in the reusable pipeline.

## How The Pipeline Works

1. Load and validate matching train and test CSV schemas.
2. Remove the activity target and subject identifier from the feature matrix.
3. Fit `StandardScaler` on the training data only.
4. Fit PCA on the training data only, then transform the test data with that same PCA object.
5. Fit one diagonal-covariance Gaussian HMM per activity. Training sequences are contiguous subject/activity runs, and their lengths are passed to `hmmlearn` so transitions are learned within sequences.
6. Classify test windows by the largest HMM log likelihood.
7. Fit Logistic Regression on the same transformed features as a transparent comparison baseline.
8. Write accuracy, macro F1, weighted F1, and class-level metrics to a JSON file.

## Local Setup

The commands below use Anaconda on Windows:

```powershell
conda create -n human-activity-hmm python=3.10 -y
conda activate human-activity-hmm
python -m pip install -r requirements.txt
```

Run the reusable pipeline:

```powershell
python scripts/run_activity_recognition.py `
  --train Data/train.csv `
  --test Data/test.csv `
  --output artifacts/metrics.json
```

Run the smoke tests without the full dataset:

```powershell
python -m pytest -q
```

The original notebook, `Human Activity Recognition from Smart Phone Data.ipynb`, is retained as a historical reference. The script under `scripts/` and the package under `src/` are the reproducible path for new runs.

## Historical Notebook Results

The original saved notebook reported approximately 0.873 HMM micro F1 at one hidden state and approximately 0.93 test F1 for its neural-network comparison. These values are preserved for context only. They are not fresh results from the reusable pipeline, and the old notebook contains a confusing early PCA cell that fits the test data separately.

## Project Limitations

- The benchmark contains engineered sensor windows rather than raw accelerometer streams.
- An HMM is meaningful only when rows are in temporal order and sequence boundaries are valid.
- The reusable implementation assumes contiguous rows for a subject represent temporal order.
- The dataset is a benchmark and should not be treated as validated performance for a new device, subject population, or safety-critical product.
- The current pipeline predicts activity classes; it is not a deployed real-time mobile application.

## Suggested Repository Name

`human-activity-hmm-benchmark` is more concise and explains both the subject and the primary method. The current GitHub repository name is unchanged until explicitly approved.
