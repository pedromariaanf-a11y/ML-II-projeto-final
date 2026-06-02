# DECISIONS.md

This file records important technical and methodological decisions that affect reproducibility, interpretation, or later modeling.

## 2026-05-30 - Use `customer_info` As The Authoritative Customer Universe

Decision: Use `customer_info.csv` as the base table for all customer-level feature engineering and future clustering outputs.

Reason: The final project must assign a cluster to every customer. `customer_info.csv` contains the full customer base, while `customer_basket.csv` contains sampled transactions for only part of that base.

Alternatives considered: Use basket customers as the base, or build the base from the union of both files.

Impact: All feature-building joins must preserve the row set from `customer_info.csv`. Basket features are enrichments, not eligibility criteria.

## 2026-05-30 - Preserve Customers Without Sampled Baskets

Decision: Keep customers who do not appear in `customer_basket.csv`.

Reason: Removing them would reduce the customer universe and violate the final requirement to provide a cluster for every `customer_id`.

Alternatives considered: Drop customers without baskets for a behavior-only segmentation.

Impact: Future clustering must account for the fact that some customers have no sampled basket behavior.

## 2026-05-30 - Use Zero Defaults For Missing Basket Aggregates

Decision: For customers without sampled baskets, set basket-derived numeric aggregates to zero, including `basket_count`, `avg_basket_size`, and `unique_basket_products`, and preserve `has_sampled_basket` as an explicit flag.

Reason: Zero defaults keep the feature table numeric and complete while the flag distinguishes missing sampled behavior from observed low activity.

Alternatives considered: Leave basket aggregates missing and impute later, or drop basket features for customers without baskets.

Impact: Clustering interpretation must consider `has_sampled_basket` so zero basket values are not mistaken for true complete inactivity.

## 2026-05-30 - Use Median Imputation For Initial Numeric Missing Values

Decision: Fill missing numeric customer-level values with each column median and add missingness indicators where missing values occurred.

Reason: Median imputation is robust to skewed customer spending and produces a complete modeling table for the first clustering experiments.

Alternatives considered: Mean imputation, zero imputation, model-based imputation, or dropping incomplete rows.

Impact: The feature table has no missing values, but imputation strategy should be reviewed before final modeling.

## 2026-05-30 - Clip Suspicious Promotion Percentages And Keep A Flag

Decision: Create `promotion_pct_clean` by clipping promotion percentages to `[0, 1]`, while preserving `promotion_pct_suspicious` and missingness flags.

Reason: Promotion share is naturally bounded between 0 and 1. Clipping makes the feature usable without silently hiding invalid values.

Alternatives considered: Treat out-of-range values as missing, drop affected rows, or use the raw values directly.

Impact: Future modeling can use the cleaned promotion feature while retaining a data-quality signal.

## 2026-05-30 - Treat Future First-Transaction Years As Invalid For Tenure

Decision: Future `year_first_transaction` values are flagged with `first_transaction_year_suspicious` and excluded from direct tenure calculation before imputation.

Reason: A future first transaction year cannot represent real historical tenure as of the analysis date.

Alternatives considered: Cap future years to the current year, keep raw values, or drop affected customers.

Impact: `customer_tenure_years` remains interpretable, and suspicious future-year records remain visible through flags.

## 2026-05-30 - Exclude Raw Identifiers From Modeling Features

Decision: Do not use raw identifiers or identifier-like fields such as `customer_name`, `loyalty_card_number`, raw `customer_birthdate`, or `customer_id` as clustering features.

Reason: These fields are identifiers or raw representations rather than meaningful behavioral dimensions. `customer_id` is retained only as a join/output key.

Alternatives considered: Encode names or loyalty card numbers directly.

Impact: The feature table remains focused on interpretable customer attributes and behavior.

## 2026-06-02 - Extract Academic Degree Indicators From Customer Name Prefix

Decision: Extract `degree_bsc`, `degree_msc`, `degree_phd`, and `degree_unknown` from the prefix of `customer_name`, while keeping the raw `customer_name` excluded from the feature table.

Reason: The assignment states that customer name contains degree-level information. The raw name is not appropriate for modeling, but the degree prefix is a clear, interpretable signal that can be reviewed as a candidate modeling or profiling feature.

Alternatives considered: Drop `customer_name` without preserving the degree signal, or encode the raw name directly.

Impact: The feature table keeps the useful academic degree signal as simple numeric flags without adding raw names to the modeling data.

## 2026-05-30 - Delay Clustering Until After Feature Review

Decision: Do not train clustering models during the preprocessing and feature-engineering phase.

Reason: Feature selection, suspicious-value handling, scaling, and interpretation choices should be reviewed before model training.

Alternatives considered: Train a quick baseline clustering model immediately after feature engineering.

Impact: The next phase can focus on a cleaner, defensible clustering experiment design.
