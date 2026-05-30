"""Feature engineering functions for the customer segmentation project."""

import ast
from collections import Counter

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


def build_customer_feature_table(customer_info, customer_basket, reference_date=None):
    """Build one clean feature table with one row per customer."""
    customer_features, imputation_values = compute_customer_features(
        customer_info, reference_date
    )

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


def compute_customer_features(customer_info, reference_date=None):
    """Create customer-level features from the full customer table."""
    customer_features = preprocess_customer_info(customer_info, reference_date)
    customer_features = drop_raw_identifier_columns(
        customer_features, RAW_NON_MODEL_COLUMNS
    )

    numeric_columns = [
        column
        for column in customer_features.select_dtypes(include=[np.number]).columns
        if column != "customer_id"
    ]
    customer_features, imputation_values = handle_missing_numeric_values(
        customer_features,
        columns=numeric_columns,
        add_indicator=True,
    )

    customer_features = compute_total_lifetime_spend(customer_features)
    customer_features = compute_spend_shares_by_category(customer_features)
    customer_features = compute_family_features(customer_features)

    return customer_features, imputation_values


def compute_total_lifetime_spend(customer_info):
    """Add total spend across the lifetime spend categories."""
    df = customer_info.copy()
    spend_values = df[SPEND_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["total_lifetime_spend"] = spend_values.sum(axis=1)
    return df


def compute_spend_shares_by_category(customer_info):
    """Add the percentage of total spend represented by each category."""
    df = customer_info.copy()
    total_spend = df["total_lifetime_spend"].replace(0, np.nan)

    for column in SPEND_COLUMNS:
        category = column.replace("lifetime_spend_", "")
        share_column = f"spend_share_{category}"
        df[share_column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        df[share_column] = (df[share_column] / total_spend).fillna(0)

    return df


def compute_family_features(customer_info):
    """Add simple household summaries from kids and teens at home."""
    df = customer_info.copy()
    kids = pd.to_numeric(df["kids_home"], errors="coerce").fillna(0)
    teens = pd.to_numeric(df["teens_home"], errors="coerce").fillna(0)

    df["total_children_home"] = kids + teens
    df["has_kids_home"] = (kids > 0).astype(int)
    df["has_teens_home"] = (teens > 0).astype(int)
    df["has_children_home"] = (df["total_children_home"] > 0).astype(int)
    return df


def parse_basket_goods(customer_basket):
    """Parse the product list string in each basket."""
    parsed_goods = []
    errors = []

    for row_index, raw_value in customer_basket["list_of_goods"].items():
        goods, error = parse_list_of_goods_safely(raw_value)
        parsed_goods.append(goods)

        if error is not None:
            errors.append(
                {"row_index": row_index, "raw_value": raw_value, "error": error}
            )

    parsed_basket = customer_basket.copy()
    parsed_basket["goods"] = parsed_goods
    parsed_basket["basket_length"] = [len(goods) for goods in parsed_goods]

    return parsed_basket, pd.DataFrame(errors)


def compute_basket_features(parsed_basket):
    """Aggregate sampled baskets to one row per customer."""
    rows = []

    for customer_id, group in parsed_basket.groupby("customer_id", sort=False):
        basket_count = int(group["invoice_id"].nunique())
        unique_products = count_unique_products(group["goods"])

        rows.append(
            {
                "customer_id": customer_id,
                "basket_count": basket_count,
                "avg_basket_size": float(group["basket_length"].mean()),
                "median_basket_size": float(group["basket_length"].median()),
                "max_basket_size": int(group["basket_length"].max()),
                "total_basket_items": int(group["basket_length"].sum()),
                "unique_basket_products": unique_products,
                "has_sampled_basket": 1,
            }
        )

    return pd.DataFrame(rows)


def merge_basket_features(customer_features, basket_features):
    """Left join basket features so customers without baskets are not dropped."""
    feature_table = customer_features.merge(basket_features, on="customer_id", how="left")

    basket_defaults = {
        "basket_count": 0,
        "avg_basket_size": 0.0,
        "median_basket_size": 0.0,
        "max_basket_size": 0,
        "total_basket_items": 0,
        "unique_basket_products": 0,
        "has_sampled_basket": 0,
    }

    for column, default_value in basket_defaults.items():
        feature_table[column] = feature_table[column].fillna(default_value)

    integer_columns = [
        "basket_count",
        "max_basket_size",
        "total_basket_items",
        "unique_basket_products",
        "has_sampled_basket",
    ]
    for column in integer_columns:
        feature_table[column] = feature_table[column].astype(int)

    return feature_table


def parse_list_of_goods_safely(value):
    """Safely parse one basket string into a list of products."""
    try:
        parsed = value if isinstance(value, list) else ast.literal_eval(value)
        if not isinstance(parsed, list):
            return [], "parsed value is not a list"
        return [str(item) for item in parsed], None
    except (SyntaxError, ValueError, TypeError) as error:
        return [], str(error)


def count_unique_products(goods_series):
    """Count unique products across all baskets for one customer."""
    product_counter = Counter()
    for goods in goods_series:
        product_counter.update(goods)
    return len(product_counter)
