# Finland Integrated Train-Weather Dataset — Delay Prediction

## Overview

This project investigates **Finnish long-distance train delays** using an integrated dataset that combines railway timetable/operational information with meteorological observations from the geographically closest Finnish Meteorological Institute (FMI) weather station.

The project uses the **2025 subset** of the Finland Integrated Train-Weather (FI-TW) dataset. The goal is to explore the data, engineer machine-learning features, and evaluate classification models for predicting whether a train event is delayed.

A row represents a single timetable event — an **arrival or departure of a long-distance train at a station** — enriched with weather observations corresponding to the event.

---

## Project Structure

### 1. Exploratory Data Analysis & Feature Engineering

**Notebook:** `Finnish_dataset_EDA_and_Feature_Engg.ipynb`

This notebook prepares the dataset for machine learning. It covers:

- Loading and combining the 2025 FI-TW data.
- Initial data inspection and exploratory analysis.
- Investigation of missing values.
- Treatment of missing weather measurements.
- Investigation of the `Snow depth = -1` sentinel value.
- Creation of the binary `delayed` target.
- Conversion of scheduled timestamps into machine-learning-friendly numerical features.
- Calendar and cyclical time feature engineering.
- Encoding of arrival/departure event type.
- Assessment and removal of `stationName` because station-level operational effects are outside the stated weather-focused scope.
- Train/test splitting before undersampling.
- Random undersampling of the training set to balance the two target classes.
- Export of the prepared datasets as:
  - `train_undersampled.csv`
  - `test.csv`

### 2. Machine Learning Evaluation

**Notebook:** `Finland_dataset_evaluation_MachineLearning.ipynb`

This notebook evaluates several classification approaches:

- Logistic Regression
- HistGradientBoosting
- Random Forest
- XGBoost
- XGBoost hyperparameter tuning using randomized and focused searches

The models are evaluated on the same untouched test set, with **ROC-AUC** used as the primary discrimination metric.

---

## Dataset

The combined dataset contains approximately **6.28 million observations and 125 columns** before the project-specific feature selection.

The original target distribution is imbalanced:

| Target | Meaning | Approx. proportion |
|---|---|---:|
| `0` | No positive delay (on time or early) | 35.9% |
| `1` | Positive delay | 64.1% |

For modelling, the data is split into training and test sets **before** undersampling. This keeps the test set representative of the original class distribution.

The resulting datasets are:

- Training set before undersampling: **4,832,366 observations**
- Training set after undersampling: **3,484,132 observations**
- Untouched test set: **1,208,092 observations**
- Training target after undersampling: **50% delayed / 50% non-delayed**
- Test target: approximately **64% delayed / 36% non-delayed**

---

## Target Variable

The binary target is derived from `differenceInMinutes`:

- `delayed = 0` → no positive delay (on time or early)
- `delayed = 1` → positive delay

Rows without a recorded delay value are removed because they cannot be assigned a target.

---

## Feature Engineering

The final modelling feature set contains 14 numeric variables:

- `month`
- `day_of_week`
- `type_encoded`
- `scheduled_time_minutes`
- `Air temperature`
- `Wind speed`
- `Snow depth`
- `Horizontal visibility`
- `sin_time`
- `cos_time`
- `sin_month`
- `cos_month`
- `sin_day`
- `cos_day`

### Time features

The original scheduled timestamp is converted into:

- Minutes after midnight.
- Calendar features such as month and day of week.
- Cyclical sine/cosine representations for time-related patterns.

The arrival/departure `type` variable is retained because the meaning of scheduled time differs between arrivals and departures.

### Station information

`stationName` contains **466 unique stations**. It is excluded from the final modelling feature set because the stated research scope focuses on the relationship between weather and delays rather than station-level operational effects.

This is a scope decision rather than a claim that station information is unimportant for prediction.

### Missing weather data

The notebooks explicitly investigate missing precipitation and snow-depth values rather than automatically treating missing observations as zero.

For snow depth, the value `-1` occurs extensively and shows a strong seasonal pattern that is consistent with its use as a sentinel for no recorded snow depth. It is therefore treated as equivalent to zero for this analysis.

---

## Train/Test Methodology

To avoid information leakage:

1. The feature matrix and target are created.
2. The data is split into training and test sets using a stratified split.
3. Only the **training set** is undersampled.
4. The test set remains untouched and retains the original class distribution.
5. All final model comparisons are performed on the same test set.

This makes the final comparison between models consistent.

---

## Model Evaluation

The final models were evaluated on the same untouched test set.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 53.84% | 0.5458 |
| HistGradientBoosting | 57.92% | 0.6051 |
| Focused XGBoost | 62.29% | 0.6722 |
| Random Forest | 69.89% | 0.7675 |

ROC-AUC is used as the primary comparison metric because it measures how well the model distinguishes delayed from non-delayed observations across classification thresholds.

The notebooks also show that XGBoost improved from a baseline ROC-AUC of approximately **0.619** to **0.672** after focused tuning and retraining on the full undersampled training set.

---

## XGBoost Tuning

Two stages of XGBoost hyperparameter search were performed.

### Randomized search

- 15 parameter combinations
- 3-fold cross-validation
- Search performed on a **300,000-row** training sample
- Best CV ROC-AUC: approximately **0.606**

### Focused search

The search space was narrowed around stronger configurations identified during the randomized search.

- 3-fold cross-validation
- Search performed on a **200,000-row** training sample
- Selected configuration included:
  - `n_estimators = 400`
  - `max_depth = 8`
  - `learning_rate = 0.10`
  - `min_child_weight = 1`

The selected configuration was then retrained on the complete undersampled training set.

---

## Model Interpretation

Feature importance was examined for the tree-based models.

For the **Random Forest**, the most influential features included:

- Air temperature
- Horizontal visibility
- Wind speed
- Scheduled time
- Time-of-day features

For **XGBoost**, the most influential features included:

- `type_encoded`
- Cyclical temporal features such as `cos_time`
- `sin_month`
- `cos_month`
- `sin_time`

These importance measures are **model-specific** and should not be interpreted as evidence of causation.

---

## How to Run

### Requirements

The notebooks use Python data-science and machine-learning libraries including:

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost

Install the required packages in your Python environment as needed, for example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Running the notebooks

Open the notebooks in Jupyter Notebook, JupyterLab, or another compatible environment.

Recommended order:

1. `Finnish_dataset_EDA_and_Feature_Engg.ipynb`
2. `Finland_dataset_evaluation_MachineLearning.ipynb`

The feature-engineering notebook prepares and exports the train/test data used by the model-evaluation notebook.

> **Note:** The original FI-TW source data is not included in this repository unless explicitly provided separately. Make sure the required input files are available at the paths expected by the notebooks before running them from scratch.

---

## Presentation

The project presentation is provided separately as a PDF/PPT export.

**Presentation PDF:**  
`[Add presentation PDF link here](path/to/presentation.pdf)`

**Presentation PowerPoint:**  
`[Add presentation PPTX link here](path/to/presentation.pptx)`

Replace the placeholder paths above with the final presentation file locations before publishing the repository.

---

## Repository Files

```text
.
├── Finnish_dataset_EDA_and_Feature_Engg.ipynb
├── Finland_dataset_evaluation_MachineLearning.ipynb
├── train_undersampled.csv          # generated by the EDA/feature-engineering notebook
├── test.csv                         # generated by the EDA/feature-engineering notebook
├── presentation.pdf                 # add final presentation PDF here
├── presentation.pptx                # add final PowerPoint here
└── README.md
```

---

## Key Takeaways

- The project combines Finnish railway and weather data to study train delays.
- The 2025 dataset contains approximately 6.28 million observations before modelling preparation.
- The modelling workflow uses careful train/test separation and training-only undersampling.
- Temporal, event-type, and weather variables are used as predictive features.
- Several linear and nonlinear classification models were evaluated.
- On the untouched test set, Random Forest achieved a ROC-AUC of **0.7675**, while the focused XGBoost model achieved **0.6722**.
- Feature importance differs between the tree-based models, highlighting that the algorithms use the available temporal, event, and weather information differently.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `Finnish_dataset_EDA_and_Feature_Engg.ipynb` | Data exploration, cleaning, feature engineering, class balancing, and dataset export |
| `Finland_dataset_evaluation_MachineLearning.ipynb` | Model training, hyperparameter tuning, evaluation, comparison, and feature interpretation |
