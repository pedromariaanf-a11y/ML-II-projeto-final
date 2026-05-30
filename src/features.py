"""Feature engineering utilities for customer segmentation."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing import (
    drop_raw_identifier_columns,
    handle_missing_numeric_values,
    preprocess_customer_info,
)


SPEND_COLUMNS = [
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


RAW_NON_MODEL_COLUMNS = [
    "customer_name",
    "customer_gender",
    "customer_birthdate",
    "customer_birthdate_parsed",
    "loyalty_card_number",
    "year_first_transaction",
    "first_transaction_year_clean",
    "percentage_of_products_bought_promotion",
    "list_of_goods",
]


def spend_columns_available(df: pd.DataFrame) -> List[str]:
    """Return lifetime spend columns present in a dataframe."""
    return [column for column in SPEND_COLUMNS if column in df.columns]


def compute_total_lifetime_spend(
    customer_info: pd.DataFrame,
    spend_columns: Optional[List[str]] = None,
    output_column: str = "total_lifetime_spend",
) -> pd.DataFrame:
    """Compute total spend across lifetime spend categories."""
    df = customer_info.copy()
    columns = spend_columns or spend_columns_available(df)
    spend_values = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[output_column] = spend_values.sum(axis=1)
    return df


def compute_spend_shares_by_category(
    customer_info: pd.DataFrame,
    spend_columns: Optional[List[str]] = None,
    total_column: str = "total_lifetime_spend",
) -> pd.DataFrame:
    """Create category spend share features from lifetime spend columns."""
    df = customer_info.copy()
    columns = spend_columns or spend_columns_available(df)

    if total_column not in df.columns:
        df = compute_total_lifetime_spend(df, columns, total_column)

    denominator = df[total_column].replace(0, np.nan)
    for column in columns:
        category = column.replace("lifetime_spend_", "")
        share_column = f"spend_share_{category}"
        df[share_column] = pd.to_numeric(df[column], errors="coerce").fillna(0) / denominator
        df[share_column] = df[share_column].fillna(0)

    return df


def compute_family_features(customer_info: pd.DataFrame) -> pd.DataFrame:
    """Create household composition features from kids and teens at home."""
    df = customer_info.copy()
    kids = pd.to_numeric(df["kids_home"], errors="coerce").fillna(0)
    teens = pd.to_numeric(df["teens_home"], errors="coerce").fillna(0)
    df["total_children_home"] = kids + teens
    df["has_kids_home"] = (kids > 0).astype(int)
    df["has_teens_home"] = (teens > 0).astype(int)
    df["has_children_home"] = (df["total_children_home"] > 0).astype(int)
    return df


def parse_list_of_goods_safely(value: Any) -> Tuple[List[str], Optional[str]]:
    """Parse one list_of_goods value without raising on malformed rows."""
    try:
        parsed = value if isinstance(value, list) else ast.literal_eval(value)
        if not isinstance(parsed, list):
            return [], "parsed value is not a list"
        return [str(item) for item in parsed], None
    except (SyntaxError, ValueError, TypeError) as exc:
        return [], str(exc)


def parse_basket_goods(
    customer_basket: pd.DataFrame,
    goods_column: str = "list_of_goods",
    parsed_column: str = "goods",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse all basket product lists and add basket length."""
    parsed_goods: List[List[str]] = []
    errors: List[Dict[str, Any]] = []

    for row_index, raw_value in customer_basket[goods_column].items():
        goods, error = parse_list_of_goods_safely(raw_value)
        parsed_goods.append(goods)
        if error is not None:
            errors.append({"row_index": row_index, "raw_value": raw_value, "error": error})

    parsed_basket = customer_basket.copy()
    parsed_basket[parsed_column] = parsed_goods
    parsed_basket["basket_length"] = [len(goods) for goods in parsed_goods]
    return parsed_basket, pd.DataFrame(errors)


def compute_basket_features(
    parsed_basket: pd.DataFrame,
    goods_column: str = "goods",
) -> pd.DataFrame:
    """Aggregate basket-derived features to one row per customer."""
    rows: List[Dict[str, Any]] = []

    grouped = parsed_basket.groupby("customer_id", sort=False)
    for customer_id, group in grouped:
        product_counter: Counter[str] = Counter()
        for goods in group[goods_column]:
            product_counter.update(goods)

        basket_count = int(group["invoice_id"].nunique())
        total_items = int(group["basket_length"].sum())

        rows.append(
            {
                "customer_id": customer_id,
                "basket_count": basket_count,
                "avg_basket_size": float(group["basket_length"].mean())
                if basket_count
                else 0.0,
                "median_basket_size": float(group["basket_length"].median())
                if basket_count
                else 0.0,
                "max_basket_size": int(group["basket_length"].max()) if basket_count else 0,
                "total_basket_items": total_items,
                "unique_basket_products": len(product_counter),
                "has_sampled_basket": 1,
            }
        )

    return pd.DataFrame(rows)


def merge_basket_features(
    customer_features: pd.DataFrame,
    basket_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join basket features onto the full customer base and fill no-basket defaults."""
    merged = customer_features.merge(basket_features, on="customer_id", how="left")

    basket_defaults = {
        "basket_count": 0,
        "avg_basket_size": 0.0,
        "median_basket_size": 0.0,
        "max_basket_size": 0,
        "total_basket_items": 0,
        "unique_basket_products": 0,
        "has_sampled_basket": 0,
    }
    for column, default in basket_defaults.items():
        if column in merged.columns:
            merged[column] = merged[column].fillna(default)

    integer_columns = [
        "basket_count",
        "max_basket_size",
        "total_basket_items",
        "unique_basket_products",
        "has_sampled_basket",
    ]
    for column in integer_columns:
        if column in merged.columns:
            merged[column] = merged[column].astype(int)

    return merged


def build_customer_feature_table(
    customer_info: pd.DataFrame,
    customer_basket: pd.DataFrame,
    reference_date: Optional[pd.Timestamp | str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a row-preserving customer-level feature table without clustering."""
    customer_features = preprocess_customer_info(customer_info, reference_date=reference_date)
    customer_features = drop_raw_identifier_columns(customer_features, RAW_NON_MODEL_COLUMNS)

    excluded_from_imputation = ["customer_id"]
    numeric_columns = [
        column
        for column in customer_features.select_dtypes(include=[np.number]).columns
        if column not in excluded_from_imputation
    ]
    customer_features, imputation_values = handle_missing_numeric_values(
        customer_features,
        columns=numeric_columns,
        strategy="median",
        add_indicator=True,
    )

    customer_features = compute_total_lifetime_spend(customer_features)
    customer_features = compute_spend_shares_by_category(customer_features)
    customer_features = compute_family_features(customer_features)

    parsed_basket, parse_errors = parse_basket_goods(customer_basket)
    basket_features = compute_basket_features(parsed_basket)
    feature_table = merge_basket_features(customer_features, basket_features)

    feature_table = feature_table.sort_values("customer_id").reset_index(drop=True)

    metadata = {
        "input_customer_count": int(customer_info["customer_id"].nunique()),
        "output_customer_count": int(feature_table["customer_id"].nunique()),
        "feature_table_shape": feature_table.shape,
        "basket_parse_errors": len(parse_errors),
        "customers_without_baskets": int((feature_table["basket_count"] == 0).sum()),
        "imputation_values": imputation_values,
        "basket_features_shape": basket_features.shape,
    }

    return feature_table, metadata
