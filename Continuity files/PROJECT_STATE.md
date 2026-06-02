# PROJECT_STATE.md

## Current State

The repository now has six completed project phases:

1. Initial repository setup and raw data audit.
2. Initial preprocessing and customer-level feature engineering.
3. Readability and defense-preparation refactor of the preprocessing workflow.
4. EDA and feature review before clustering.
5. Baseline K-Means clustering experiments.
6. Candidate cluster profiling and interpretability review.

The current pipeline builds an in-memory feature table, compares baseline K-Means clustering options, profiles selected candidate solutions, and preserves every customer from `customer_info.csv`. No final model has been selected and no final customer-cluster CSV has been created.

Continuity documents now live together in `Continuity files/`. Raw datasets and the assignment PDF now live together in `Project files/`. The data loading code supports this layout.

## Completed Setup

- Added a clean project structure with `src/`, `notebooks/`, `data/`, and `outputs/`.
- Added `requirements.txt` for notebook-based analysis.
- Added reusable data loading helpers in `src/data_loading.py`.
- Added reusable data audit helpers in `src/data_audit.py`.
- Added `notebooks/01_data_audit.ipynb` for the first dataset audit.
- Preserved raw CSV files unchanged.
- Organized continuity documents under `Continuity files/`.
- Organized raw project inputs under `Project files/`.
- Updated data loading and notebooks to find raw data in `Project files/`.

## Preprocessing Work Completed

- Added `src/preprocessing.py` with reusable functions for:
  - parsing `customer_birthdate`,
  - deriving `customer_age`,
  - deriving `customer_tenure_years`,
  - creating `has_loyalty_card`,
  - extracting academic degree indicators from the `customer_name` prefix,
  - flagging missing/invalid date and tenure values,
  - clipping suspicious promotion percentages into `promotion_pct_clean`,
  - preserving suspicious promotion/year signals as explicit flags,
  - median-imputing numeric values with missingness indicators.
- Added `src/features.py` with reusable functions for:
  - total lifetime spend,
  - spend shares by category,
  - family/household features,
  - safe parsing of `list_of_goods`,
  - customer-level basket aggregates,
  - left-joining basket features onto the full customer base.
- Added `notebooks/02_preprocessing_features.ipynb` to build and validate the customer-level feature table.

## Readability And Defense Refactor Completed

- Refactored `notebooks/02_preprocessing_features.ipynb` so the notebook is the main explanation layer.
- Added a simple step-by-step pipeline overview in the notebook.
- Added short explanations before each major step: loading data, preprocessing customer data, creating spend features, creating family features, creating basket features, merging basket features, and validating customer preservation.
- Refactored `src/preprocessing.py` and `src/features.py` so main functions are easier to identify and helper functions are separated from the project-facing workflow.
- Added clearer docstrings explaining why each main function exists.
- Preserved the existing preprocessing and feature-engineering results.

## Academic-Style Simplification Completed

- Simplified `src/data_loading.py`, `src/data_audit.py`, `src/preprocessing.py`, and `src/features.py` to look more like a notebook-based academic data science project.
- Removed advanced type-hinting style and unnecessary typing imports from the active audit, preprocessing, and feature-engineering modules.
- Removed notebook helper type hints that made the notebooks look more production-oriented than necessary.
- Kept the main workflow functions easy to explain: `preprocess_customer_info`, `compute_customer_features`, `compute_basket_features`, `merge_basket_features`, and `build_customer_feature_table`.
- Reduced helper-function complexity while keeping small helpers that make the notebook and source code easier to follow.
- Kept the notebook as the main explanation layer for defense preparation.
- Updated the feature table from 33,038 x 73 to 33,038 x 77 after adding academic degree prefix features.
- Added a small `.gitignore` to keep future Python cache, notebook checkpoint, virtual environment, and generated output files out of version control.

## EDA And Feature Review Completed

- Added `notebooks/03_eda_feature_review.ipynb` as the next project phase before clustering.
- The notebook uses `REFERENCE_DATE = "2026-05-30"` when rebuilding the feature table so age and tenure are reproducible.
- The notebook validates that the feature table has 33,038 rows x 77 columns, unique `customer_id`, no missing values, no raw `customer_name`, and all four degree features.
- The notebook reviews demographic, household, spend, basket, loyalty, complaints, promotion, degree, outlier, skewness, and correlation patterns.
- Refined the correlation/redundancy review with a readable high-correlation feature-pair table and interpretation thresholds.
- Added a manual redundancy decision table explaining which groups should be modeled, profiled, or excluded before clustering.
- Defined two explicit baseline modeling feature sets:
  - `model_features_a_no_basket`: 20 customer-info-derived features with no basket features.
  - `model_features_b_with_basket`: 23 features that add only `has_sampled_basket`, `basket_count`, and `avg_basket_size`.
- Defined `profiling_features` for interpretation and `excluded_features` for row keys/raw identifiers.
- Kept `unique_basket_products` as profiling/sensitivity for now because it is highly correlated with `basket_count`.
- Added optional matplotlib plots for readability; the notebook still relies mainly on simple tables.
- No clustering model was trained.
- No final customer-cluster CSV was created.

## Baseline Clustering Completed

- Added `notebooks/04_baseline_clustering.ipynb` as the first baseline clustering notebook.
- Added `scikit-learn>=1.3` to `requirements.txt` for K-Means, scaling, and silhouette scoring.
- The notebook rebuilds the feature table with `REFERENCE_DATE = "2026-05-30"` and validates the same 33,038 rows x 77 columns.
- K-Means was tested with `RANDOM_STATE = 42` for `k = 2` through `k = 10`.
- Baseline A was evaluated with 20 customer-info-derived features and no basket features.
- Baseline B was evaluated with 23 features: Baseline A plus `has_sampled_basket`, `basket_count`, and `avg_basket_size`.
- For each feature set and `k`, the notebook records inertia, silhouette score, cluster counts, smallest/largest cluster size, and smallest/largest cluster percentage.
- Cluster-size summaries and preliminary interpretation checks were added for promising candidates.
- No final clustering model has been selected.
- No final customer-cluster CSV was created.

## Candidate Cluster Profiling Completed

- Added `src/modeling.py` with small reusable modeling utilities for transformations, scaling, K-Means labels, in-memory cluster labels, cluster sizes, cluster profiles, and comparisons with global averages.
- Added `notebooks/05_candidate_cluster_profiling.ipynb` to profile selected K-Means candidate solutions before final model selection.
- The notebook rebuilds the feature table with `REFERENCE_DATE = "2026-05-30"` and validates the same 33,038 rows x 77 columns.
- Profiled five candidate solutions: Baseline A with `k = 2`, `k = 4`, and `k = 6`; Baseline B with `k = 4` and `k = 6`.
- Cluster size summaries, cluster profile tables, differences from global averages, and strongest distinguishing features were created in memory.
- Added basket availability diagnostics for Baseline B candidates to check whether clusters are too basket-driven.
- The notebook recommends `A_k4` or `A_k6`, plus `B_k4` if basket diagnostics remain interpretable, as candidates to investigate further.
- No final clustering model has been selected.
- No final customer-cluster CSV was created.

## Feature Table Status

- Feature table shape is now 33,038 rows x 77 columns.
- One row is present for every one of the 33,038 customers in `customer_info.csv`.
- `customer_id` remains unique.
- Customers without sampled baskets are retained.
- 4,911 customers have `basket_count = 0`, `avg_basket_size = 0`, and `unique_basket_products = 0`.
- No missing values remain after preprocessing.
- Raw `customer_name` remains excluded from the feature table, but its academic degree prefix is preserved as clean numeric flags.
- Baseline clustering experiments and candidate profiling have been performed, but no final clustering model has been selected.
- No final customer segment CSV has been created.
- No report or application has been created.

## Key Features Created

- Demographic: `customer_age`, gender indicator features, degree prefix indicator features.
- Tenure: `customer_tenure_years`, first-transaction quality flags.
- Loyalty and complaints: `has_loyalty_card`, `loyalty_card_missing`, `number_complaints`.
- Promotion: `promotion_pct_clean`, `promotion_pct_suspicious`, `promotion_pct_missing`.
- Family: `total_children_home`, `has_kids_home`, `has_teens_home`, `has_children_home`.
- Spend: `total_lifetime_spend` and spend-share features by category.
- Basket behavior: `basket_count`, `avg_basket_size`, `median_basket_size`, `max_basket_size`, `total_basket_items`, `unique_basket_products`, `has_sampled_basket`.

## Known Data Notes Found So Far

- `Project files/customer_info.csv` has 33,038 rows and 25 columns.
- `Project files/customer_basket.csv` has 100,000 rows and 3 columns.
- Every customer ID present in `customer_basket.csv` exists in `customer_info.csv`.
- No duplicate rows, duplicate customer IDs, or duplicate invoice IDs were found.
- `list_of_goods` parsed successfully for all 100,000 baskets.
- Basket lengths range from 1 to 18 products, with a median of 9 products.
- `loyalty_card_number` has 13,106 missing values (39.67%).
- The audit flagged 1,755 promotion percentages outside `[0, 1]`; these are currently clipped to 0 or 1 and flagged.
- The audit flagged 991 first-transaction years outside `[1900, current_year]`; these are currently treated as invalid for tenure and flagged.
- Academic degree prefix features currently identify 5,154 BSc customers, 5,057 MSc customers, 5,096 PhD customers, and 17,731 customers without one of those known prefixes.

## Unresolved Data Quality Decisions

- Decide whether negative promotion percentages should be clipped to 0, treated as missing, or investigated as a data-generation issue.
- Decide whether future first-transaction years should be imputed, capped, excluded from tenure, or used as a synthetic-data signal.
- Decide whether median imputation is appropriate for all customer numeric fields before clustering.
- Decide whether geographic coordinates should be transformed into distance/location-cluster features before modeling.
- Decide whether raw spend magnitudes, spend shares, or both should be used in clustering after scaling.

## Known Risks

- Basket data is sampled, so basket-derived features may underrepresent some customers.
- Customers without baskets have zero basket features; clustering should avoid interpreting those zeros as identical to true low engagement without considering `has_sampled_basket`.
- High-spend features may dominate distance-based clustering unless scaled or transformed.
- Missingness indicators may become useful synthetic-data signals, but they can also bias cluster interpretation.
- The project should not advance too quickly into clustering before the project owner can understand and defend the feature table.
- The workflow must remain understandable and defensible, not only technically functional.

## Current Task

Review the candidate cluster profiling notebook and decide which candidate solution should move toward final model selection.

## Next Recommended Task

Select one candidate configuration for final validation, document the reasoning, and only then create the final customer-cluster output.

## Known Issues

- No final clustering workflow exists yet.
- No final customer-cluster CSV exists yet.
- No report or application exists yet.
- A final model-selection decision still needs to be made.
