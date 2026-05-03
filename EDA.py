from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("customer_info.csv")
DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

# Fixed project reference date so age and tenure are reproducible.
REFERENCE_DATE = pd.Timestamp("2026-05-03")
REFERENCE_YEAR = REFERENCE_DATE.year

SPEND_COLS = [
    "lifetime_spend_groceries",
    "lifetime_spend_electronics",
    "lifetime_spend_vegetables",
    "lifetime_spend_nonalcohol_drinks",
    "lifetime_spend_alcohol_drinks",
    "lifetime_spend_meat",
    "lifetime_spend_fish",
    "lifetime_spend_hygiene",
    "lifetime_spend_videogames",
    "lifetime_spend_petfood",
]

INTEGER_COLS = [
    "kids_home",
    "teens_home",
    "number_complaints",
    "distinct_stores_visited",
    "typical_hour",
    "lifetime_total_distinct_products",
    "year_first_transaction",
]

PROMO_COL = "percentage_of_products_bought_promotion"


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{value:.2f}%"


def money_like_name(column: str) -> str:
    return (
        column.replace("lifetime_spend_", "")
        .replace("nonalcohol_drinks", "non-alcohol drinks")
        .replace("_", " ")
        .title()
    )


def calculate_age(dates: pd.Series) -> pd.Series:
    birthday_passed = (
        (REFERENCE_DATE.month > dates.dt.month)
        | ((REFERENCE_DATE.month == dates.dt.month) & (REFERENCE_DATE.day >= dates.dt.day))
    )
    return REFERENCE_DATE.year - dates.dt.year - (~birthday_passed).astype("int64")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_missing_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns and df[column].isna().any():
            df[f"{column}_was_missing"] = df[column].isna().astype("int64")
    return df


def impute_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            continue
        median = df[column].median(skipna=True)
        if pd.isna(median):
            median = 0
        df[column] = df[column].fillna(median)
    return df


def normalize_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [column.strip().lower() for column in df.columns]
    df["customer_gender"] = df["customer_gender"].str.strip().str.lower()

    original_missing_cols = [column for column in df.columns if df[column].isna().any()]
    df = add_missing_flags(df, original_missing_cols)

    birthdate = pd.to_datetime(
        df["customer_birthdate"],
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    df["customer_birthdate"] = birthdate.dt.strftime("%Y-%m-%d").fillna("unknown")
    df["age"] = calculate_age(birthdate)
    df["birthdate_parse_failed"] = birthdate.isna().astype("int64")

    # This column contains 1 when present and missing otherwise, so treat it as
    # a membership flag instead of a numeric card number.
    df["has_loyalty_card"] = df["loyalty_card_number"].notna().astype("int64")
    df = df.drop(columns=["loyalty_card_number"])

    promo_invalid = df[PROMO_COL].lt(0) | df[PROMO_COL].gt(1)
    df["promotion_rate_invalid"] = promo_invalid.fillna(False).astype("int64")
    df.loc[promo_invalid, PROMO_COL] = np.nan

    future_transaction_year = df["year_first_transaction"].gt(REFERENCE_YEAR)
    df["future_first_transaction_year"] = future_transaction_year.fillna(False).astype("int64")
    df.loc[future_transaction_year, "year_first_transaction"] = np.nan

    impute_cols = (
        SPEND_COLS
        + INTEGER_COLS
        + [PROMO_COL, "age", "latitude", "longitude"]
    )
    df = impute_numeric(df, impute_cols)

    for column in INTEGER_COLS:
        df[column] = df[column].round().astype("int64")

    df["age"] = df["age"].round().astype("int64")
    df[PROMO_COL] = df[PROMO_COL].clip(0, 1)

    df["total_lifetime_spend"] = df[SPEND_COLS].sum(axis=1)
    df["household_children"] = df["kids_home"] + df["teens_home"]
    df["customer_tenure_years"] = REFERENCE_YEAR - df["year_first_transaction"]
    df["avg_spend_per_product"] = safe_divide(
        df["total_lifetime_spend"],
        df["lifetime_total_distinct_products"],
    )
    df["spend_per_tenure_year"] = safe_divide(
        df["total_lifetime_spend"],
        df["customer_tenure_years"].clip(lower=1),
    )

    df["avg_spend_per_product"] = df["avg_spend_per_product"].fillna(
        df["avg_spend_per_product"].median()
    )
    df["spend_per_tenure_year"] = df["spend_per_tenure_year"].fillna(
        df["spend_per_tenure_year"].median()
    )

    hour_radians = 2 * np.pi * df["typical_hour"] / 24
    df["typical_hour_sin"] = np.sin(hour_radians)
    df["typical_hour_cos"] = np.cos(hour_radians)

    for column in SPEND_COLS:
        suffix = column.replace("lifetime_spend_", "")
        df[f"share_spend_{suffix}"] = safe_divide(
            df[column],
            df["total_lifetime_spend"],
        ).fillna(0)
        df[f"log1p_{column}"] = np.log1p(df[column])

    return df


def make_feature_sets(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gender_male = cleaned["customer_gender"].eq("male").astype("int64")

    feature_df = pd.DataFrame(
        {
            "customer_id": cleaned["customer_id"],
            "age": cleaned["age"],
            "gender_male": gender_male,
            "kids_home": cleaned["kids_home"],
            "teens_home": cleaned["teens_home"],
            "household_children": cleaned["household_children"],
            "number_complaints": cleaned["number_complaints"],
            "distinct_stores_visited": cleaned["distinct_stores_visited"],
            "customer_tenure_years": cleaned["customer_tenure_years"],
            "has_loyalty_card": cleaned["has_loyalty_card"],
            "promotion_purchase_rate": cleaned[PROMO_COL],
            "lifetime_total_distinct_products": cleaned["lifetime_total_distinct_products"],
            "total_lifetime_spend": cleaned["total_lifetime_spend"],
            "avg_spend_per_product": cleaned["avg_spend_per_product"],
            "spend_per_tenure_year": cleaned["spend_per_tenure_year"],
            "typical_hour_sin": cleaned["typical_hour_sin"],
            "typical_hour_cos": cleaned["typical_hour_cos"],
            "latitude": cleaned["latitude"],
            "longitude": cleaned["longitude"],
        }
    )

    for column in SPEND_COLS:
        suffix = column.replace("lifetime_spend_", "")
        feature_df[f"log1p_spend_{suffix}"] = cleaned[f"log1p_{column}"]
        feature_df[f"share_spend_{suffix}"] = cleaned[f"share_spend_{suffix}"]

    feature_columns = [column for column in feature_df.columns if column != "customer_id"]
    scaling = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean": feature_df[feature_columns].mean().values,
            "std": feature_df[feature_columns].std(ddof=0).replace(0, 1).values,
        }
    )

    scaled = feature_df[["customer_id"]].copy()
    for _, row in scaling.iterrows():
        feature = row["feature"]
        scaled[feature] = (feature_df[feature] - row["mean"]) / row["std"]

    return feature_df, scaled, scaling


def raw_quality_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = (
        raw.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(
            missing_pct=lambda frame: frame["missing_count"] / len(raw) * 100,
            dtype=[str(dtype) for dtype in raw.dtypes],
        )
        .sort_values(["missing_count", "missing_pct"], ascending=False)
    )

    numeric_summary = raw.describe().T.reset_index().rename(columns={"index": "column"})
    categorical_summary = raw.describe(include=["object", "string"]).T.reset_index()
    categorical_summary = categorical_summary.rename(columns={"index": "column"})

    invalid = pd.DataFrame(
        [
            {
                "check": "duplicate_customer_id",
                "affected_rows": int(raw["customer_id"].duplicated().sum()),
                "action": "No action required because all ids are unique.",
            },
            {
                "check": "duplicate_rows",
                "affected_rows": int(raw.duplicated().sum()),
                "action": "No action required because no full-row duplicates were found.",
            },
            {
                "check": "negative_promotion_rate",
                "affected_rows": int(raw[PROMO_COL].lt(0).sum()),
                "action": "Set to missing, then median-imputed after adding an invalid-value flag.",
            },
            {
                "check": f"first_transaction_after_{REFERENCE_YEAR}",
                "affected_rows": int(raw["year_first_transaction"].gt(REFERENCE_YEAR).sum()),
                "action": "Set to missing, then median-imputed after adding an invalid-value flag.",
            },
            {
                "check": "loyalty_card_number_missing",
                "affected_rows": int(raw["loyalty_card_number"].isna().sum()),
                "action": "Converted to has_loyalty_card = 0; present values converted to 1.",
            },
        ]
    )

    return {
        "missing_values": missing.reset_index().rename(columns={"index": "column"}),
        "numeric_summary_raw": numeric_summary,
        "categorical_summary_raw": categorical_summary,
        "invalid_value_checks": invalid,
    }


def cleaned_summary_tables(
    cleaned: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    spend_summary = (
        cleaned[SPEND_COLS]
        .agg(["mean", "median", "std", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "column"})
    )
    spend_summary.insert(1, "category", spend_summary["column"].map(money_like_name))
    spend_summary = spend_summary.sort_values("mean", ascending=False)

    core_numeric = [
        "age",
        "customer_tenure_years",
        "kids_home",
        "teens_home",
        "number_complaints",
        "distinct_stores_visited",
        "lifetime_total_distinct_products",
        PROMO_COL,
        "total_lifetime_spend",
        "avg_spend_per_product",
        "spend_per_tenure_year",
        "latitude",
        "longitude",
    ]

    cleaned_numeric_summary = (
        cleaned[core_numeric]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T.reset_index()
        .rename(columns={"index": "column"})
    )

    feature_summary = (
        features.drop(columns=["customer_id"])
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )

    corr = features.drop(columns=["customer_id"]).corr(numeric_only=True)
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
    corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False)

    return {
        "spend_summary_cleaned": spend_summary,
        "cleaned_numeric_summary": cleaned_numeric_summary,
        "model_feature_summary": feature_summary,
        "model_feature_correlations": corr.reset_index().rename(columns={"index": "feature"}),
        "top_absolute_correlations": corr_pairs.head(30),
    }


def svg_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path: Path, body: str, width: int, height: int) -> None:
    path.write_text(
        dedent(
            f"""\
            <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
              <rect width="100%" height="100%" fill="#ffffff"/>
              {body}
            </svg>
            """
        ),
        encoding="utf-8",
    )


def write_bar_chart(
    series: pd.Series,
    path: Path,
    title: str,
    value_suffix: str = "",
    width: int = 980,
    bar_height: int = 28,
) -> None:
    data = series.dropna().astype(float)
    if data.empty:
        return

    label_width = 270
    left = 330
    right = 80
    top = 78
    gap = 12
    height = top + len(data) * (bar_height + gap) + 60
    max_value = max(data.max(), 1)
    plot_width = width - left - right

    parts = [
        f'<text x="32" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#18202a">{svg_escape(title)}</text>'
    ]
    for index, (label, value) in enumerate(data.items()):
        y = top + index * (bar_height + gap)
        bar_width = max(2, value / max_value * plot_width)
        parts.extend(
            [
                f'<text x="{label_width}" y="{y + 20}" text-anchor="end" font-family="Arial" font-size="14" fill="#344054">{svg_escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="4" fill="#2f6f73"/>',
                f'<text x="{left + bar_width + 8:.1f}" y="{y + 20}" font-family="Arial" font-size="13" fill="#344054">{value:.2f}{value_suffix}</text>',
            ]
        )

    write_svg(path, "\n".join(parts), width, height)


def write_histogram(
    values: pd.Series,
    path: Path,
    title: str,
    x_label: str,
    bins: int = 24,
    width: int = 980,
    height: int = 520,
) -> None:
    clean_values = values.dropna().astype(float)
    if clean_values.empty:
        return

    counts, edges = np.histogram(clean_values, bins=bins)
    left = 80
    right = 45
    bottom = 78
    top = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(counts.max(), 1)
    bar_gap = 2
    bar_width = plot_width / bins - bar_gap

    parts = [
        f'<text x="32" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#18202a">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#98a2b3" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#98a2b3" stroke-width="1"/>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14" fill="#344054">{svg_escape(x_label)}</text>',
        f'<text x="22" y="{top - 15}" font-family="Arial" font-size="13" fill="#667085">Count</text>',
    ]

    for index, count in enumerate(counts):
        x = left + index * (plot_width / bins)
        bar_h = count / max_count * plot_height
        y = height - bottom - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="#6a994e"/>'
        )

    for tick in np.linspace(edges[0], edges[-1], 5):
        x = left + (tick - edges[0]) / (edges[-1] - edges[0]) * plot_width
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{height - bottom}" x2="{x:.1f}" y2="{height - bottom + 5}" stroke="#98a2b3"/>',
                f'<text x="{x:.1f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#667085">{tick:.0f}</text>',
            ]
        )

    write_svg(path, "\n".join(parts), width, height)


def corr_color(value: float) -> str:
    value = max(-1.0, min(1.0, float(value)))
    if value >= 0:
        low = np.array([245, 247, 250])
        high = np.array([47, 111, 115])
        rgb = low + (high - low) * value
    else:
        low = np.array([245, 247, 250])
        high = np.array([174, 62, 74])
        rgb = low + (high - low) * abs(value)
    return "#" + "".join(f"{int(channel):02x}" for channel in rgb)


def write_correlation_heatmap(corr: pd.DataFrame, path: Path, title: str) -> None:
    labels = list(corr.columns)
    cell = 34
    left = 230
    top = 180
    width = left + len(labels) * cell + 70
    height = top + len(labels) * cell + 60

    parts = [
        f'<text x="32" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#18202a">{svg_escape(title)}</text>',
        '<text x="32" y="70" font-family="Arial" font-size="13" fill="#667085">Positive correlations are teal; negative correlations are red.</text>',
    ]

    for i, label in enumerate(labels):
        x = left + i * cell + cell / 2
        y = top - 8
        parts.append(
            f'<text x="{x:.1f}" y="{y}" transform="rotate(-55 {x:.1f} {y})" text-anchor="start" font-family="Arial" font-size="11" fill="#344054">{svg_escape(label)}</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{top + i * cell + 22}" text-anchor="end" font-family="Arial" font-size="11" fill="#344054">{svg_escape(label)}</text>'
        )

    for row_i, row_label in enumerate(labels):
        for col_i, col_label in enumerate(labels):
            value = corr.loc[row_label, col_label]
            x = left + col_i * cell
            y = top + row_i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{corr_color(value)}" stroke="#ffffff" stroke-width="1"/>'
            )

    write_svg(path, "\n".join(parts), width, height)


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value).replace("|", "\\|")

    headers = [str(column).replace("|", "\\|") for column in df.columns]
    rows = [
        [format_value(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def build_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    features: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> str:
    missing = tables["missing_values"]
    invalid = tables["invalid_value_checks"]
    spend = tables["spend_summary_cleaned"]
    top_corr = tables["top_absolute_correlations"]

    missing_columns = int((missing["missing_count"] > 0).sum())
    total_missing_cells = int(missing["missing_count"].sum())
    duplicate_ids = int(invalid.loc[invalid["check"].eq("duplicate_customer_id"), "affected_rows"].iloc[0])
    negative_promos = int(invalid.loc[invalid["check"].eq("negative_promotion_rate"), "affected_rows"].iloc[0])
    future_years = int(invalid.loc[invalid["check"].str.startswith("first_transaction_after"), "affected_rows"].iloc[0])
    loyalty_missing = int(invalid.loc[invalid["check"].eq("loyalty_card_number_missing"), "affected_rows"].iloc[0])

    age = cleaned["age"]
    tenure = cleaned["customer_tenure_years"]
    total_spend = cleaned["total_lifetime_spend"]
    loyalty_rate = cleaned["has_loyalty_card"].mean() * 100

    top_missing_md = markdown_table(missing.head(8))
    top_spend_md = markdown_table(spend[["category", "mean", "median", "max"]].head(10))
    top_corr_md = markdown_table(
        top_corr[["feature_a", "feature_b", "correlation"]].head(12),
        floatfmt=".3f",
    )

    report = f"""# Customer Info Cleaning and EDA

Reference date: {REFERENCE_DATE.date()}

## Dataset Snapshot

- Raw shape: {raw.shape[0]:,} rows x {raw.shape[1]:,} columns.
- Cleaned shape: {cleaned.shape[0]:,} rows x {cleaned.shape[1]:,} columns.
- Model feature matrix: {features.shape[0]:,} rows x {features.shape[1] - 1:,} numeric features plus `customer_id`.
- Duplicate customer ids: {duplicate_ids:,}.
- Total missing raw cells: {total_missing_cells:,} across {missing_columns:,} columns.

## Cleaning and Format Adjustments Applied

- Parsed `customer_birthdate` from `MM/DD/YYYY HH:MM AM/PM` into ISO date strings and derived `age`.
- Standardized `customer_gender` values to lowercase labels.
- Converted integer-like float columns such as kids, teens, complaints, stores, typical hour, products, and first transaction year back to integer values after imputation.
- Converted `loyalty_card_number` into `has_loyalty_card`, because present values are always `1` and missing values indicate no card.
- Marked negative promotion rates as invalid, replaced them with missing values, and median-imputed them.
- Marked first transaction years after {REFERENCE_YEAR} as invalid, replaced them with missing values, and median-imputed them.
- Added missing-value and invalid-value flags in the cleaned dataset for auditability.
- Engineered clustering-friendly variables: age, tenure, total lifetime spend, spend per tenure year, average spend per product, household children, cyclical typical hour, log spend features, and spend-share features.

## Raw Data Quality

- Missing `loyalty_card_number`: {loyalty_missing:,} rows ({loyalty_missing / len(raw) * 100:.2f}%).
- Invalid negative promotion rates: {negative_promos:,} rows ({negative_promos / len(raw) * 100:.2f}%).
- Future first transaction years beyond {REFERENCE_YEAR}: {future_years:,} rows ({future_years / len(raw) * 100:.2f}%).
- Birthdate parse failures or missing dates: {int(cleaned["birthdate_parse_failed"].sum()):,} rows.

Top missing columns:

{top_missing_md}

## Customer Profile

- Age range after cleaning: {age.min():.0f} to {age.max():.0f}; median age is {age.median():.0f}.
- Tenure range after cleaning: {tenure.min():.0f} to {tenure.max():.0f} years; median tenure is {tenure.median():.0f}.
- Loyalty-card coverage after conversion: {loyalty_rate:.2f}%.
- Median total lifetime spend: {total_spend.median():,.2f}; mean total lifetime spend: {total_spend.mean():,.2f}.

## Spending Overview

{top_spend_md}

Groceries dominate absolute spending, so clustering should avoid feeding raw spend columns only. The exported feature file includes log-transformed spend and category share features to support both value-based and preference-based clustering.

## Strongest Feature Correlations

{top_corr_md}

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
"""
    return report


def make_figures(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    features: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    missing = tables["missing_values"].copy()
    missing = missing.loc[missing["missing_count"] > 0].head(16)
    missing_series = missing.set_index("column")["missing_pct"].sort_values(ascending=True)
    write_bar_chart(
        missing_series,
        FIGURE_DIR / "missing_values.svg",
        "Raw Missing Values by Column",
        value_suffix="%",
    )

    spend = tables["spend_summary_cleaned"].set_index("category")["mean"].sort_values()
    write_bar_chart(
        spend,
        FIGURE_DIR / "mean_spend_by_category.svg",
        "Mean Lifetime Spend by Category",
    )

    write_histogram(
        cleaned["age"],
        FIGURE_DIR / "age_distribution.svg",
        "Customer Age Distribution",
        "Age",
    )
    write_histogram(
        cleaned["customer_tenure_years"],
        FIGURE_DIR / "tenure_distribution.svg",
        "Customer Tenure Distribution",
        "Tenure years",
        bins=20,
    )

    heatmap_cols = [
        "age",
        "customer_tenure_years",
        "household_children",
        "number_complaints",
        "distinct_stores_visited",
        "promotion_purchase_rate",
        "lifetime_total_distinct_products",
        "total_lifetime_spend",
        "avg_spend_per_product",
        "spend_per_tenure_year",
        "has_loyalty_card",
        "gender_male",
    ]
    corr = features[heatmap_cols].corr(numeric_only=True)
    write_correlation_heatmap(
        corr,
        FIGURE_DIR / "correlation_heatmap.svg",
        "Correlation Heatmap for Core Model Features",
    )


def main() -> None:
    ensure_dirs()

    raw = pd.read_csv(RAW_DATA_PATH)
    cleaned = normalize_raw_data(raw)
    features, scaled_features, scaling = make_feature_sets(cleaned)

    raw_tables = raw_quality_tables(raw)
    cleaned_tables = cleaned_summary_tables(cleaned, features)
    tables = {**raw_tables, **cleaned_tables, "scaling_parameters": scaling}

    cleaned.to_csv(DATA_DIR / "customer_info_cleaned.csv", index=False)
    features.to_csv(DATA_DIR / "customer_info_model_features.csv", index=False)
    scaled_features.to_csv(DATA_DIR / "customer_info_model_features_scaled.csv", index=False)
    write_tables(tables)
    make_figures(raw, cleaned, features, tables)

    report = build_report(raw, cleaned, features, tables)
    (REPORT_DIR / "eda_report.md").write_text(report, encoding="utf-8")

    print("Cleaning and EDA complete.")
    print(f"Cleaned data: {DATA_DIR / 'customer_info_cleaned.csv'}")
    print(f"Model features: {DATA_DIR / 'customer_info_model_features.csv'}")
    print(f"Scaled features: {DATA_DIR / 'customer_info_model_features_scaled.csv'}")
    print(f"EDA report: {REPORT_DIR / 'eda_report.md'}")


if __name__ == "__main__":
    main()
