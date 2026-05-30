"""Preprocessing functions used before feature engineering.

The goal is to make raw customer columns usable for analysis while keeping the
steps easy to explain in the notebook.
"""

import numpy as np
import pandas as pd


RAW_IDENTIFIER_COLUMNS = [
    "customer_name",
    "customer_birthdate",
    "customer_birthdate_parsed",
    "loyalty_card_number",
]


def preprocess_customer_info(customer_info, reference_date=None):
    """Create clean customer fields from the raw customer table."""
    df = parse_customer_birthdate(customer_info)
    df = derive_customer_age(df, reference_date)
    df = derive_customer_tenure(df, reference_date)
    df = create_has_loyalty_card(df)
    df = handle_suspicious_promotion_percentages(df)
    df = add_gender_features(df)
    return df


def parse_customer_birthdate(customer_info):
    """Parse birthdate strings and flag missing or unparseable values."""
    df = customer_info.copy()
    parsed_birthdate = parse_dates(df["customer_birthdate"])

    df["customer_birthdate_parsed"] = parsed_birthdate
    df["customer_birthdate_missing"] = df["customer_birthdate"].isna().astype(int)
    df["customer_birthdate_parse_failed"] = (
        df["customer_birthdate"].notna() & parsed_birthdate.isna()
    ).astype(int)
    return df


def derive_customer_age(customer_info, reference_date=None):
    """Convert parsed birthdates into age in years."""
    df = customer_info.copy()
    reference = get_reference_date(reference_date)

    age = (reference - df["customer_birthdate_parsed"]).dt.days / 365.25
    suspicious_age = age.notna() & ~age.between(0, 110)

    df["customer_age_suspicious"] = suspicious_age.astype(int)
    df["customer_age"] = age.mask(suspicious_age)
    df["customer_age_missing_or_invalid"] = df["customer_age"].isna().astype(int)
    return df


def derive_customer_tenure(customer_info, reference_date=None):
    """Convert first transaction year into tenure in years."""
    df = customer_info.copy()
    reference_year = get_reference_date(reference_date).year

    first_year = pd.to_numeric(df["year_first_transaction"], errors="coerce")
    suspicious_year = first_year.notna() & ~first_year.between(1900, reference_year)

    df["first_transaction_year_suspicious"] = suspicious_year.astype(int)
    df["first_transaction_year_clean"] = first_year.mask(suspicious_year)
    df["customer_tenure_years"] = reference_year - df["first_transaction_year_clean"]
    df["customer_tenure_missing_or_invalid"] = (
        df["customer_tenure_years"].isna().astype(int)
    )
    return df


def create_has_loyalty_card(customer_info):
    """Replace the raw loyalty card number with a yes/no feature."""
    df = customer_info.copy()
    loyalty_number = pd.to_numeric(df["loyalty_card_number"], errors="coerce")

    df["has_loyalty_card"] = loyalty_number.notna().astype(int)
    df["loyalty_card_missing"] = loyalty_number.isna().astype(int)
    return df


def handle_suspicious_promotion_percentages(customer_info):
    """Create a clean promotion percentage and keep flags for bad values."""
    df = customer_info.copy()
    promotion_pct = pd.to_numeric(
        df["percentage_of_products_bought_promotion"], errors="coerce"
    )

    df["promotion_pct_suspicious"] = (
        promotion_pct.notna() & ~promotion_pct.between(0, 1)
    ).astype(int)
    df["promotion_pct_clean"] = promotion_pct.clip(lower=0, upper=1)
    df["promotion_pct_missing"] = promotion_pct.isna().astype(int)
    return df


def add_gender_features(customer_info):
    """Create simple numeric indicators from the gender label."""
    df = customer_info.copy()
    gender = df["customer_gender"].astype("string").str.lower().str.strip()

    df["gender_female"] = (gender == "female").astype(int)
    df["gender_male"] = (gender == "male").astype(int)
    df["gender_unknown"] = (~gender.isin(["female", "male"])).astype(int)
    return df


def handle_missing_numeric_values(df, columns=None, add_indicator=True):
    """Fill missing numeric values with the median and keep missing flags."""
    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns

    imputation_values = {}
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")

        if add_indicator and values.isna().any():
            result[f"{column}_was_missing"] = values.isna().astype(int)

        fill_value = values.median() if values.notna().any() else 0
        result[column] = values.fillna(fill_value)
        imputation_values[column] = float(fill_value)

    return result, imputation_values


def drop_raw_identifier_columns(df, columns=None):
    """Remove columns that are identifiers rather than modelling features."""
    columns_to_drop = columns if columns is not None else RAW_IDENTIFIER_COLUMNS
    existing_columns = [column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns)


def get_reference_date(reference_date=None):
    """Use a fixed date when provided; otherwise use today's date."""
    if reference_date is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(reference_date).normalize()


def parse_dates(values):
    """Parse mixed date strings while staying compatible with older pandas."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")
