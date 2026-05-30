"""Preprocessing utilities for customer-level segmentation features."""

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


def _reference_timestamp(reference_date: Optional[pd.Timestamp | str] = None) -> pd.Timestamp:
    if reference_date is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(reference_date).normalize()


def _datetime_from_mixed_formats(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def parse_customer_birthdate(
    customer_info: pd.DataFrame,
    source_column: str = "customer_birthdate",
    output_column: str = "customer_birthdate_parsed",
) -> pd.DataFrame:
    """Parse the raw customer birthdate column and flag failed parses."""
    df = customer_info.copy()
    parsed = _datetime_from_mixed_formats(df[source_column])
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
    """Derive customer age and flag ages outside a plausible range."""
    df = customer_info.copy()
    reference = _reference_timestamp(reference_date)
    age = (reference - df[parsed_column]).dt.days / 365.25
    suspicious = age.notna() & ~age.between(min_age, max_age)
    df["customer_age_suspicious"] = suspicious.astype(int)
    df[output_column] = age.mask(suspicious)
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
    """Derive tenure from the first transaction year and flag future years."""
    df = customer_info.copy()
    reference_year = _reference_timestamp(reference_date).year
    year = pd.to_numeric(df[source_column], errors="coerce")
    suspicious = year.notna() & ~year.between(min_year, reference_year)
    df["first_transaction_year_suspicious"] = suspicious.astype(int)
    df[clean_year_column] = year.mask(suspicious)
    df[output_column] = reference_year - df[clean_year_column]
    df["customer_tenure_missing_or_invalid"] = df[output_column].isna().astype(int)
    return df


def create_has_loyalty_card(
    customer_info: pd.DataFrame,
    source_column: str = "loyalty_card_number",
    output_column: str = "has_loyalty_card",
) -> pd.DataFrame:
    """Convert the raw loyalty-card number into a binary feature."""
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
    """Clip promotion percentages to [0, 1] and keep an explicit suspicious flag."""
    df = customer_info.copy()
    promo = pd.to_numeric(df[source_column], errors="coerce")
    suspicious = promo.notna() & ~promo.between(0, 1)
    df["promotion_pct_suspicious"] = suspicious.astype(int)
    df[output_column] = promo.clip(lower=0, upper=1)
    df["promotion_pct_missing"] = promo.isna().astype(int)
    return df


def add_gender_features(
    customer_info: pd.DataFrame, source_column: str = "customer_gender"
) -> pd.DataFrame:
    """Create numeric gender indicators without using the raw text as a model feature."""
    df = customer_info.copy()
    gender = df[source_column].astype("string").str.lower().str.strip()
    df["gender_female"] = (gender == "female").astype(int)
    df["gender_male"] = (gender == "male").astype(int)
    df["gender_unknown"] = (~gender.isin(["female", "male"])).astype(int)
    return df


def numeric_columns_for_imputation(
    df: pd.DataFrame, exclude_columns: Optional[Iterable[str]] = None
) -> List[str]:
    """Return numeric columns suitable for missing-value handling."""
    excluded = set(exclude_columns or [])
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def handle_missing_numeric_values(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, float]] = None,
    add_indicator: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fill missing numeric values and optionally add missingness indicators."""
    result = df.copy()
    target_columns = list(columns) if columns is not None else numeric_columns_for_imputation(result)
    imputation_values: Dict[str, float] = {}

    for column in target_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        has_missing = bool(values.isna().any())
        if add_indicator and has_missing:
            result[f"{column}_was_missing"] = values.isna().astype(int)

        if fill_values and column in fill_values:
            fill_value = float(fill_values[column])
        elif strategy == "median":
            fill_value = float(values.median()) if values.notna().any() else 0.0
        elif strategy == "mean":
            fill_value = float(values.mean()) if values.notna().any() else 0.0
        elif strategy == "zero":
            fill_value = 0.0
        else:
            raise ValueError(f"Unsupported numeric imputation strategy: {strategy}")

        result[column] = values.fillna(fill_value)
        imputation_values[column] = fill_value

    return result, imputation_values


def preprocess_customer_info(
    customer_info: pd.DataFrame,
    reference_date: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Apply the customer-level preprocessing steps needed before feature engineering."""
    df = parse_customer_birthdate(customer_info)
    df = derive_customer_age(df, reference_date=reference_date)
    df = derive_customer_tenure(df, reference_date=reference_date)
    df = create_has_loyalty_card(df)
    df = handle_suspicious_promotion_percentages(df)
    df = add_gender_features(df)
    return df


def drop_raw_identifier_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Drop raw identifier-like columns that should not be modeling features."""
    columns_to_drop = list(columns or RAW_IDENTIFIER_COLUMNS)
    return df.drop(columns=[column for column in columns_to_drop if column in df.columns])
