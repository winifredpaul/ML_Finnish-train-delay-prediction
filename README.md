# Raw Data

The raw data used in this project comes from the **Finland Integrated Train-Weather Dataset (FI-TW)**.

## Source

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/viniborin/finland-integrated-train-weather-dataset-fi-tw

The original dataset contains monthly train and weather data covering multiple years.

## Project Scope

For this project, we use **only the 2025 data** from the original dataset. The 2025 subset consists of 12 monthly Parquet files, from January through December.

We selected a single year to keep the dataset size manageable while providing sufficient data for developing and evaluating a machine learning model for train delay prediction.

## Files Used

- `matched_data_flat_2025_01.parquet`
- `matched_data_flat_2025_02.parquet`
- `matched_data_flat_2025_03.parquet`
- `matched_data_flat_2025_04.parquet`
- `matched_data_flat_2025_05.parquet`
- `matched_data_flat_2025_06.parquet`
- `matched_data_flat_2025_07.parquet`
- `matched_data_flat_2025_08.parquet`
- `matched_data_flat_2025_09.parquet`
- `matched_data_flat_2025_10.parquet`
- `matched_data_flat_2025_11.parquet`
- `matched_data_flat_2025_12.parquet`

The original Parquet files are not stored in this GitHub repository because they are relatively large. They can be downloaded from the Kaggle source above and placed in this `data/raw/` directory before running the notebook.

## Data Processing

The raw data will be cleaned and transformed during the project. The resulting dataset used for machine learning will be stored separately under:

`data/processed/`
