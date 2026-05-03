# Customer Info Cleaning and EDA

Reference date: 2026-05-03

## Dataset Snapshot

- Raw shape: 33,038 rows x 25 columns.
- Cleaned shape: 33,038 rows x 72 columns.
- Model feature matrix: 33,038 rows x 38 numeric features plus `customer_id`.
- Duplicate customer ids: 0.
- Total missing raw cells: 20,869 across 16 columns.

## Cleaning and Format Adjustments Applied

- Parsed `customer_birthdate` from `MM/DD/YYYY HH:MM AM/PM` into ISO date strings and derived `age`.
- Standardized `customer_gender` values to lowercase labels.
- Converted integer-like float columns such as kids, teens, complaints, stores, typical hour, products, and first transaction year back to integer values after imputation.
- Converted `loyalty_card_number` into `has_loyalty_card`, because present values are always `1` and missing values indicate no card.
- Marked negative promotion rates as invalid, replaced them with missing values, and median-imputed them.
- Marked first transaction years after 2026 as invalid, replaced them with missing values, and median-imputed them.
- Added missing-value and invalid-value flags in the cleaned dataset for auditability.
- Engineered clustering-friendly variables: age, tenure, total lifetime spend, spend per tenure year, average spend per product, household children, cyclical typical hour, log spend features, and spend-share features.

## Raw Data Quality

- Missing `loyalty_card_number`: 13,106 rows (39.67%).
- Invalid negative promotion rates: 1,755 rows (5.31%).
- Future first transaction years beyond 2026: 991 rows (3.00%).
- Birthdate parse failures or missing dates: 165 rows.

Top missing columns:

| column | missing_count | missing_pct | dtype |
| --- | --- | --- | --- |
| loyalty_card_number | 13106 | 39.67 | float64 |
| lifetime_spend_fish | 991 | 3.00 | float64 |
| number_complaints | 661 | 2.00 | float64 |
| lifetime_spend_electronics | 661 | 2.00 | float64 |
| typical_hour | 661 | 2.00 | float64 |
| lifetime_spend_vegetables | 661 | 2.00 | float64 |
| lifetime_spend_meat | 661 | 2.00 | float64 |
| lifetime_spend_videogames | 661 | 2.00 | float64 |

## Customer Profile

- Age range after cleaning: 24 to 86; median age is 54.
- Tenure range after cleaning: 0 to 33 years; median tenure is 11.
- Loyalty-card coverage after conversion: 60.33%.
- Median total lifetime spend: 20,269.00; mean total lifetime spend: 23,706.24.

## Spending Overview

| category | mean | median | max |
| --- | --- | --- | --- |
| Groceries | 16306.23 | 13002.50 | 104670.00 |
| Electronics | 2737.21 | 1470.00 | 35299.00 |
| Hygiene | 819.00 | 686.00 | 3482.00 |
| Meat | 723.83 | 729.00 | 3052.00 |
| Vegetables | 722.10 | 471.00 | 3337.00 |
| Alcohol Drinks | 620.75 | 483.00 | 3704.00 |
| Fish | 605.85 | 511.00 | 3172.00 |
| Non-Alcohol Drinks | 464.35 | 421.00 | 2180.00 |
| Videogames | 370.88 | 223.00 | 3936.00 |
| Petfood | 336.03 | 327.00 | 1224.00 |

Groceries dominate absolute spending, so clustering should avoid feeding raw spend columns only. The exported feature file includes log-transformed spend and category share features to support both value-based and preference-based clustering.

## Strongest Feature Correlations

| feature_a | feature_b | correlation |
| --- | --- | --- |
| kids_home | household_children | 0.871 |
| teens_home | household_children | 0.809 |
| log1p_spend_groceries | share_spend_groceries | 0.798 |
| share_spend_groceries | share_spend_electronics | -0.783 |
| total_lifetime_spend | log1p_spend_groceries | 0.758 |
| log1p_spend_electronics | share_spend_electronics | 0.715 |
| log1p_spend_videogames | share_spend_videogames | 0.677 |
| log1p_spend_alcohol_drinks | share_spend_alcohol_drinks | 0.664 |
| log1p_spend_vegetables | share_spend_vegetables | 0.662 |
| log1p_spend_fish | share_spend_fish | 0.645 |
| share_spend_vegetables | log1p_spend_meat | -0.645 |
| log1p_spend_meat | log1p_spend_fish | 0.643 |

High correlations are expected between total spend, spend per tenure year, average spend per product, and category-level spend features. For K-Means or hierarchical clustering, start with a smaller feature subset and compare results with and without redundant spend metrics.

## Figures

- `reports/figures/missing_values.svg`
- `reports/figures/mean_spend_by_category.svg`
- `reports/figures/age_distribution.svg`
- `reports/figures/tenure_distribution.svg`
- `reports/figures/correlation_heatmap.svg`

## Output Files

- `data/customer_info_cleaned.csv`: cleaned and audited dataset with engineered fields.
- `data/customer_info_model_features.csv`: numeric, no-missing feature matrix for unsupervised learning.
- `data/customer_info_model_features_scaled.csv`: z-score scaled version for distance-based clustering and PCA.
- `reports/tables/*.csv`: reusable EDA tables and scaling parameters.

## Recommended Next Steps for Unsupervised Learning

1. Start with `customer_info_model_features_scaled.csv`.
2. Compare at least two feature subsets: customer value features and category preference/share features.
3. Use PCA or UMAP for visual inspection before committing to cluster counts.
4. For K-Means, test several `k` values with inertia and silhouette score.
5. Profile clusters back on `customer_info_cleaned.csv` so the final interpretation is in original business units.
