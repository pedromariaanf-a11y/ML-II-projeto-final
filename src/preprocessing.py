"""Preprocessing utilities for customer-level segmentation features.

The functions in this module prepare raw customer fields for later clustering.
They do not choose clustering features or train models.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RAW_IDENTIFIER_COLUMNS = [
    "customer_name",
    "customer_birthdate",
    "customer_birthdate_parsed",
    "loyalty_card_number",
]


# ---------------------------------------------------------------------------
# Main preprocessing functions
# ---------------------------------------------------------------------------


def preprocess_customer_info(
    customer_info: pd.DataFrame,
    reference_date: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Create customer-level preprocessing fields before feature engineering.

    This keeps the main customer table intact while converting raw dates,
    loyalty-card values, promotion percentages, and gender labels into numeric
    fields that are easier to explain and eventually model.
    """
    df = parse_customer_birthdate(customer_info)
    df = derive_customer_age(df, reference_date=reference_date)
    df = derive_customer_tenure(df, reference_date=reference_date)
    df = create_has_loyalty_card(df)
    df = handle_suspicious_promotion_percentages(df)
    df = add_gender_features(df)
    return df


def parse_customer_birthdate(
    customer_info: pd.DataFrame,
    source_column: str = "customer_birthdate",
    output_column: str = "customer_birthdate_parsed",
) -> pd.DataFrame:
    """Parse raw birthdate strings and keep flags for missing or failed values."""
    df = customer_info.copy()
    parsed = _parse_mixed_datetime(df[source_column])

    df[output_column] = parsed
    df["customer_birthdate_missing"] = df[source_column].isna().astype(int)
    df["customer_birthdate_parse_failed"] = (
        df[source_column].notna() & parsed.isna()
    ).astype(int)
    return df


def derive_customer_age(
    customer_info: pd.DataFrame,
    reference_date: Optional[pd.Timestamp | str] = None,
    parsed_column: str = "customer_birthdate_parsed",
    output_column: str = "customer_age",
    min_age: int = 0,
    max_age: int = 110,
) -> pd.DataFrame:
    """Convert parsed birthdates into age and flag implausible ages."""
    df = customer_info.copy()
    reference = _reference_timestamp(reference_date)

    age = (reference - df[parsed_column]).dt.days / 365.25
    suspicious_age = age.notna() & ~age.between(min_age, max_age)

    df["customer_age_suspicious"] = suspicious_age.astype(int)
    df[output_column] = age.mask(suspicious_age)
    df["customer_age_missing_or_invalid"] = df[output_column].isna().astype(int)
    return df


def derive_customer_tenure(
    customer_info: pd.DataFrame,
    reference_date: Optional[pd.Timestamp | str] = None,
    source_column: str = "year_first_transaction",
    clean_year_column: str = "first_transaction_year_clean",
    output_column: str = "customer_tenure_years",
    min_year: int = 1900,
) -> pd.DataFrame:
    """Convert first transaction year into customer tenure.

    Future years are not silently used as negative tenure. They are set aside
    through a clean year column and preserved through a quality flag.
    """
    df = customer_info.copy()
    reference_year = _reference_timestamp(reference_date).year

    year = pd.to_numeric(df[source_column], errors="coerce")
    suspicious_year = year.notna() & ~year.between(min_year, reference_year)

    df["first_transaction_year_suspicious"] = suspicious_year.astype(int)
    df[clean_year_column] = year.mask(suspicious_year)
    df[output_column] = reference_year - df[clean_year_column]
    df["customer_tenure_missing_or_invalid"] = df[output_column].isna().astype(int)
    return df


def create_has_loyalty_card(
    customer_info: pd.DataFrame,
    source_column: str = "loyalty_card_number",
    output_column: str = "has_loyalty_card",
) -> pd.DataFrame:
    """Replace raw loyalty-card numbers with an interpretable yes/no feature."""
    df = customer_info.copy()
    loyalty_value = pd.to_numeric(df[source_column], errors="coerce")

    df[output_column] = loyalty_value.notna().astype(int)
    df["loyalty_card_missing"] = loyalty_value.isna().astype(int)
    return df


def handle_suspicious_promotion_percentages(
    customer_info: pd.DataFrame,
    source_column: str = "percentage_of_products_bought_promotion",
    output_column: str = "promotion_pct_clean",
) -> pd.DataFrame:
    """Create a bounded promotion percentage and keep data-quality flags."""
    df = customer_info.copy()
    promotion_pct = pd.to_numeric(df[source_column], errors="coerce")
    suspicious_pct = promotion_pct.notna() & ~promotion_pct.between(0, 1)

    df["promotion_pct_suspicious"] = suspicious_pct.astype(int)
    df[output_column] = promotion_pct.clip(lower=0, upper=1)
    df["promotion_pct_missing"] = promotion_pct.isna().astype(int)
    return df


def add_gender_features(
    customer_info: pd.DataFrame,
    source_column: str = "customer_gender",
) -> pd.DataFrame:
    """Turn the raw gender label into simple indicator columns."""
    df = customer_info.copy()
    gender = df[source_column].astype("string").str.lower().str.strip()

    df["gender_female"] = (gender == "female").astype(int)
    df["gender_male"] = (gender == "male").astype(int)
    df["gender_unknown"] = (~gender.isin(["female", "male"])).astype(int)
    return df


def handle_missing_numeric_values(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, float]] = None,
    add_indicator: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fill numeric missing values while preserving missingness as a signal.

    Median is the current default because several customer spending fields are
    skewed. Indicators are only created for columns that actually have missing
    values, keeping the final table easier to read.
    """
    result = df.copy()
    target_columns = (
        list(columns)
        if columns is not None
        else _numeric_columns_for_imputation(result)
    )
    imputation_values: Dict[str, float] = {}

    for column in target_columns:
        values = pd.to_numeric(result[column], errors="coerce")

        if add_indicator and values.isna().any():
            result[f"{column}_was_missing"] = values.isna().astype(int)

        fill_value = _choose_numeric_fill_value(
            values=values,
            column=column,
            strategy=strategy,
            fill_values=fill_values,
        )
        result[column] = values.fillna(fill_value)
        imputation_values[column] = fill_value

    return result, imputation_values


def drop_raw_identifier_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Remove raw identifier-like fields before building modeling features."""
    columns_to_drop = list(columns or RAW_IDENTIFIER_COLUMNS)
    present_columns = [column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=present_columns)


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------


def _reference_timestamp(reference_date: Optional[pd.Timestamp | str] = None) -> pd.Timestamp:
    if reference_date is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(reference_date).normalize()


def _parse_mixed_datetime(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _numeric_columns_for_imputation(
    df: pd.DataFrame,
    exclude_columns: Optional[Iterable[str]] = None,
) -> List[str]:
    excluded = set(exclude_columns or [])
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def _choose_numeric_fill_value(
    values: pd.Series,
    column: str,
    strategy: str,
    fill_values: Optional[Dict[str, float]] = None,
) -> float:
    if fill_values and column in fill_values:
        return float(fill_values[column])
    if strategy == "median":
        return float(values.median()) if values.notna().any() else 0.0
    if strategy == "mean":
        return float(values.mean()) if values.notna().any() else 0.0
    if strategy == "zero":
        return 0.0
    raise ValueError(f"Unsupported numeric imputation strategy: {strategy}")
