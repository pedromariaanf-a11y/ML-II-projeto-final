# PROJECT_STATE.md

## Current State

The repository now has two completed project phases:

1. Initial repository setup and raw data audit.
2. Initial preprocessing and customer-level feature engineering.

The current pipeline builds an in-memory feature table for future clustering. It preserves every customer from `customer_info.csv`, adds optional basket-derived behavior features from `customer_basket.csv`, and does not train clustering models or create final cluster outputs.

## Completed Setup

- Added a clean project structure with `src/`, `notebooks/`, `data/`, and `outputs/`.
- Added `requirements.txt` for notebook-based analysis.
- Added reusable data loading helpers in `src/data_loading.py`.
- Added reusable data audit helpers in `src/data_audit.py`.
- Added `notebooks/01_data_audit.ipynb` for the first dataset audit.
- Preserved raw CSV files unchanged.

## Preprocessing Work Completed

- Added `src/preprocessing.py` with reusable functions for:
  - parsing `customer_birthdate`,
  - deriving `customer_age`,
  - deriving `customer_tenure_years`,
  - creating `has_loyalty_card`,
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

## Feature Table Status

- Feature table shape: 33,038 rows x 73 columns.
- One row is present for every `customer_id` in `customer_info.csv`.
- `customer_id` remains unique.
- Customers without sampled baskets are retained.
- 4,911 customers have `basket_count = 0`, `avg_basket_size = 0`, and `unique_basket_products = 0`.
- No missing values remain after preprocessing.
- No clustering has been performed.
- No final customer segment CSV has been created.

## Key Features Created

- Demographic: `customer_age`, gender indicator features.
- Tenure: `customer_tenure_years`, first-transaction quality flags.
- Loyalty and complaints: `has_loyalty_card`, `loyalty_card_missing`, `number_complaints`.
- Promotion: `promotion_pct_clean`, `promotion_pct_suspicious`, `promotion_pct_missing`.
- Family: `total_children_home`, `has_kids_home`, `has_teens_home`, `has_children_home`.
- Spend: `total_lifetime_spend` and spend-share features by category.
- Basket behavior: `basket_count`, `avg_basket_size`, `median_basket_size`, `max_basket_size`, `total_basket_items`, `unique_basket_products`, `has_sampled_basket`.

## Known Data Notes Found So Far

- `customer_info.csv` has 33,038 rows and 25 columns.
- `customer_basket.csv` has 100,000 rows and 3 columns.
- Every customer ID present in `customer_basket.csv` exists in `customer_info.csv`.
- No duplicate rows, duplicate customer IDs, or duplicate invoice IDs were found.
- `list_of_goods` parsed successfully for all 100,000 baskets.
- Basket lengths range from 1 to 18 products, with a median of 9 products.
- `loyalty_card_number` has 13,106 missing values (39.67%).
- The audit flagged 1,755 promotion percentages outside `[0, 1]`; these are currently clipped to 0 or 1 and flagged.
- The audit flagged 991 first-transaction years outside `[1900, current_year]`; these are currently treated as invalid for tenure and flagged.

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

## Current Task

Review the preprocessing feature table and select a modeling-ready feature subset plus scaling strategy.

## Next Recommended Task

Implement the first clustering experiment workflow: choose candidate numeric features, scale/transform them, compare a small set of clustering methods and cluster counts, and evaluate cluster quality without creating the final submission CSV yet.

## Known Issues

- No clustering workflow exists yet.
- No final customer-cluster CSV exists yet.
- No report or application exists yet.
- Modeling feature selection and scaling decisions still need to be made.
