# PROJECT_STATE.md

## Current State

The repository has been initialized for the first data audit phase. It contains raw input data, minimal project documentation, reusable loading/audit modules, and a notebook that audits the available datasets without performing clustering.

## Completed Setup

- Added a clean project structure with `src/`, `notebooks/`, `data/`, and `outputs/`.
- Added `requirements.txt` for the first audit workflow.
- Added reusable data loading helpers in `src/data_loading.py`.
- Added reusable data audit helpers in `src/data_audit.py`.
- Added `notebooks/01_data_audit.ipynb` to run the first dataset audit.
- Preserved raw CSV files unchanged.

## Known Data Notes Found So Far

- `customer_info.csv` has 33,038 rows and 25 columns.
- `customer_basket.csv` has 100,000 rows and 3 columns.
- Every customer ID present in `customer_basket.csv` exists in `customer_info.csv`.
- 4,911 customers in `customer_info.csv` do not appear in the sampled basket data, so basket features cannot be required for all customers.
- No duplicate rows, duplicate customer IDs, or duplicate invoice IDs were found.
- `list_of_goods` parsed successfully for all 100,000 baskets.
- Basket lengths range from 1 to 18 products, with a median of 9 products.
- `loyalty_card_number` has 13,106 missing values (39.67%).
- Several spending and behavior fields contain smaller missingness rates between 0.5% and 3.0%.
- The audit flagged 1,755 `percentage_of_products_bought_promotion` values outside `[0, 1]`.
- The audit flagged 991 `year_first_transaction` values outside `[1900, current_year]`.

## Current Task

Review the first data audit findings and decide preprocessing rules for missing values and suspicious ranges.

## Next Recommended Task

Create a baseline feature engineering workflow that combines customer demographics, spending behavior, loyalty/complaint signals, geography, and optional basket-derived aggregates into a customer-level modeling table.

## Known Issues

- No clustering workflow exists yet.
- No final customer-cluster CSV exists yet.
- No report or application exists yet.
- Preprocessing decisions for missing values, dates, categorical variables, and basket features still need to be made.
