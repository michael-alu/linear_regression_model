# AI Job Market Salary Predictor

## Demo Video

https://www.canva.com/design/DAHE4Z65v50/nxuRkVJM-qphum8L5I5j6w/watch?utm_content=DAHE4Z65v50&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=ha106bf629a

## Description

This project predicts AI job salaries (in USD) using a multivariate linear regression model trained on 15,000 records from the [Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025) dataset. It provides a publicly accessible REST API and a Flutter mobile app that consumes the API, enabling aspiring AI professionals to make informed, data-driven career and salary decisions.

## Live API

**Base URL:** https://linear-regression-api-3o1x.onrender.com

**Swagger Documentation:** https://linear-regression-api-3o1x.onrender.com/docs

## Repo Structure

```
linear_regression_model/
├── README.md
├── .gitignore
│
└── summative/
    ├── linear_regression/            # Jupyter notebook & model artifacts
    │   ├── multivariate.ipynb        # EDA, feature engineering & model training
    │   ├── ai_job_dataset.csv        # Raw dataset (15 000 rows)
    │   ├── best_model.pkl            # Trained Ridge regression model
    │   └── scaler.pkl                # StandardScaler fitted on training data
    │
    ├── API/                          # FastAPI prediction service
    │   ├── prediction.py             # API entry point (POST /predict)
    │   ├── feature_columns.json      # Ordered feature column names
    │   ├── requirements.txt          # Python dependencies
    │   └── req.rest                  # Sample REST Client requests
    │
    └── FlutterApp/                   # Flutter mobile client
        ├── lib/
        │   └── main.dart             # App entry point & prediction UI
        ├── pubspec.yaml              # Flutter dependencies
        ├── android/                  # Android platform files
        ├── ios/                      # iOS platform files
        ├── web/                      # Web platform files
        ├── linux/                    # Linux platform files
        ├── macos/                    # macOS platform files
        └── windows/                  # Windows platform files
```

## How to Run the Flutter App

### Prerequisites

| Tool                        | Version | Install guide                                |
| --------------------------- | ------- | -------------------------------------------- |
| Flutter SDK                 | ≥ 3.11  | https://docs.flutter.dev/get-started/install |
| Android Studio **or** Xcode | Latest  | Required for Android / iOS emulators         |
| Chrome                      | Latest  | Required only for `flutter run -d chrome`    |

> After installing Flutter, run `flutter doctor` to verify your setup.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/michael-alu/linear_regression_model.git
cd linear_regression_model/summative/FlutterApp

# 2. Install dependencies
flutter pub get

# 3. Run the app (pick ONE target)

# — Android emulator (must be running)
flutter run

# — iOS simulator (macOS only)
flutter run -d ios

# — Chrome browser
flutter run -d chrome

# — macOS desktop (macOS only)
flutter run -d macos
```

> **Note:**

The app connects to the live API at `https://linear-regression-api-3o1x.onrender.com`. No local API setup is required.

## Dataset

**Source:**

[Kaggle — Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)

**Author:**

Bisma Sajjad

**Size:**

15,000 rows × 19 original columns, expanded to 61 features after feature engineering.
