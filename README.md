# Customer Segmentation Project

## Objective

Build a reproducible customer segmentation solution for the Machine Learning II assignment. The final project should identify meaningful customer groups, explain their behavior, propose targeted marketing campaigns, and eventually export a CSV with every `customer_id` assigned to a final cluster.

At the current stage, preprocessing, feature engineering, EDA, baseline clustering experiments, and candidate profiling have been completed. No final model has been selected and no final customer-cluster CSV has been created.

## Dataset Description

The raw datasets and original assignment brief are stored in `Project files/`:

- `Project files/customer_info.csv`: one row per customer with demographics, household information, complaints, loyalty-card status, lifetime spending by category, location, distinct products, and first transaction year.
- `Project files/customer_basket.csv`: sampled shopping baskets with `invoice_id`, `customer_id`, and `list_of_goods` stored as a Python-style list string.
- `Project files/Machine Learning II - Project Statement (1).pdf`: the assignment statement.

Raw CSV files must remain unchanged. The data loader prefers `Project files/` and raises an error if duplicate raw CSV copies are found in both the repository root and `Project files/`.

## Setup Instructions

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Recommended Notebook Order

Run notebooks from the repository root, in this order:

1. `notebooks/01_data_audit.ipynb`: audits the raw datasets, schema, data quality, basket parsing, and customer ID overlap.
2. `notebooks/02_preprocessing_features.ipynb`: builds the customer-level feature table with a fixed reference date for reproducible age and tenure.
3. `notebooks/03_eda_feature_review.ipynb`: reviews the engineered feature table and defines baseline feature sets.
4. `notebooks/04_baseline_clustering.ipynb`: compares baseline K-Means options for Baseline A and Baseline B. This is exploratory and does not select a final model.
5. `notebooks/05_candidate_cluster_profiling.ipynb`: profiles selected candidate clustering solutions and keeps Baseline B as a diagnostic/sensitivity option.

Example command-line validation:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda_feature_review.ipynb
```

## Current Feature Table

The current feature-building workflow creates an in-memory feature table with:

- 33,038 customers.
- 77 columns after academic degree indicators are extracted from `customer_name`.
- Unique `customer_id`.
- No missing values after preprocessing.
- Raw `customer_name` excluded from modeling data.

The final customer-cluster output has intentionally not been created yet.

## Repository Structure

```text
.
|-- requirements.txt
|-- README.md
|-- Continuity files/
|   |-- AGENTS.md
|   |-- DECISIONS.md
|   `-- PROJECT_STATE.md
|-- Project files/
|   |-- Machine Learning II - Project Statement (1).pdf
|   |-- customer_basket.csv
|   `-- customer_info.csv
|-- data/
|   `-- .gitkeep
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_preprocessing_features.ipynb
|   |-- 03_eda_feature_review.ipynb
|   |-- 04_baseline_clustering.ipynb
|   `-- 05_candidate_cluster_profiling.ipynb
|-- outputs/
|   `-- .gitkeep
`-- src/
    |-- data_audit.py
    |-- data_loading.py
    |-- features.py
    |-- modeling.py
    `-- preprocessing.py
```
