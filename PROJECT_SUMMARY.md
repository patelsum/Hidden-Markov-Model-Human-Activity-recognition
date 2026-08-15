# Project Summary: Human Activity Recognition With HMMs

Status: In progress

## What The Project Does

The project identifies a person’s activity from smartphone sensor measurements. It recognizes laying, sitting, standing, walking, walking downstairs, and walking upstairs.

## How It Works

The reusable pipeline reads the UCI Human Activity Recognition train and test CSV files, removes the target and subject identifier, scales the features, and applies PCA. It then trains one Gaussian Hidden Markov Model for each activity. Test windows are scored by each model and assigned to the activity with the highest likelihood. Logistic Regression is trained on the same transformed features as a simple comparison baseline.

## Data Summary

The original notebook contains saved outputs for 7,352 training rows and 2,947 test rows. It reports six classes and 562 input columns before preprocessing. The raw CSV files are not stored in this public repository; reviewers download them from the documented Kaggle source.

## Key Improvements

- Added a reproducible command-line pipeline.
- Added explicit dependency and test configuration.
- Corrected the reusable preprocessing design so PCA is fit on training data only.
- Excluded `subject` from predictive features and documented that decision.
- Passed contiguous sequence lengths into the HMM instead of treating every row as an independent one-row sequence.
- Added a Logistic Regression baseline without requiring TensorFlow.
- Added privacy-safe data and artifact ignore rules.

## Test Status

The repository includes a synthetic-data smoke test that exercises preprocessing, HMM fitting, prediction, and metric generation without committing the benchmark dataset. A full benchmark run requires the external CSV files and the documented environment.

## Reviewer Takeaway

This is an educational comparison of temporal probabilistic modelling and a conventional classifier. The important modelling constraint is that an HMM needs ordered sequences; the project therefore documents sequence assumptions instead of presenting a one-row likelihood classifier as full temporal modelling.
