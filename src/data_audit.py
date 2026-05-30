"""Reusable audit helpers for the customer segmentation datasets."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return column-level type and completeness information."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().to_numpy(),
            "missing_count": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean().to_numpy() * 100).round(2),
        }
    )


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize columns with missing values."""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean().to_numpy() * 100).round(2),
        }
    )
    return summary.loc[summary["missing_count"] > 0].sort_values(
        ["missing_count", "column"], ascending=[False, True]
    )


def duplicate_summary(
    customer_info: pd.DataFrame, customer_basket: pd.DataFrame
) -> pd.DataFrame:
    """Report duplicate rows and duplicate key fields."""
    rows = [
        {
            "dataset": "customer_info",
            "check": "duplicate_rows",
            "count": int(customer_info.duplicated().sum()),
        },
        {
            "dataset": "customer_info",
            "check": "duplicate_customer_id",
            "count": int(customer_info["customer_id"].duplicated().sum()),
        },
        {
            "dataset": "customer_basket",
            "check": "duplicate_rows",
            "count": int(customer_basket.duplicated().sum()),
        },
        {
            "dataset": "customer_basket",
            "check": "duplicate_invoice_id",
            "count": int(customer_basket["invoice_id"].duplicated().sum()),
        },
    ]
    return pd.DataFrame(rows)


def validate_customer_overlap(
    customer_info: pd.DataFrame, customer_basket: pd.DataFrame, sample_size: int = 10
) -> Dict[str, Any]:
    """Compare customer IDs between the customer and basket datasets."""
    info_ids = set(customer_info["customer_id"].dropna())
    basket_ids = set(customer_basket["customer_id"].dropna())

    basket_ids_missing_in_info = sorted(basket_ids - info_ids)
    customers_without_baskets = sorted(info_ids - basket_ids)

    return {
        "customer_info_unique_customers": len(info_ids),
        "customer_basket_unique_customers": len(basket_ids),
        "basket_ids_missing_in_info": len(basket_ids_missing_in_info),
        "sample_basket_ids_missing_in_info": basket_ids_missing_in_info[:sample_size],
        "customers_without_baskets": len(customers_without_baskets),
        "customers_without_baskets_pct": round(
            len(customers_without_baskets) / len(info_ids) * 100, 2
        )
        if info_ids
        else np.nan,
        "sample_customers_without_baskets": customers_without_baskets[:sample_size],
    }


def _parse_goods(value: Any) -> List[str]:
    if isinstance(value, list):
        parsed = value
    else:
        parsed = ast.literal_eval(value)

    if not isinstance(parsed, list):
        raise ValueError("list_of_goods value is not a list")

    return [str(item) for item in parsed]


def parse_goods_column(
    customer_basket: pd.DataFrame,
    goods_column: str = "list_of_goods",
    parsed_column: str = "goods",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse basket product lists and add basket length."""
    parsed_goods: List[List[str]] = []
    errors: List[Dict[str, Any]] = []

    for row_index, value in customer_basket[goods_column].items():
        try:
            parsed_goods.append(_parse_goods(value))
        except (SyntaxError, ValueError, TypeError) as exc:
            parsed_goods.append([])
            errors.append(
                {
                    "row_index": row_index,
                    "raw_value": value,
                    "error": str(exc),
                }
            )

    parsed_basket = customer_basket.copy()
    parsed_basket[parsed_column] = parsed_goods
    parsed_basket["basket_length"] = [len(goods) for goods in parsed_goods]

    return parsed_basket, pd.DataFrame(errors)


def basket_length_distribution(parsed_basket: pd.DataFrame) -> pd.DataFrame:
    """Summarize parsed basket lengths."""
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return (
        parsed_basket["basket_length"]
        .describe(percentiles=percentiles)
        .rename("basket_length")
        .reset_index()
        .rename(columns={"index": "metric"})
    )


def top_products(
    parsed_basket: pd.DataFrame, goods_column: str = "goods", top_n: int = 20
) -> pd.DataFrame:
    """Return the most frequent products across parsed baskets."""
    product_counts: Counter[str] = Counter()
    for goods in parsed_basket[goods_column]:
        product_counts.update(goods)

    total_baskets = len(parsed_basket)
    rows = [
        {
            "product": product,
            "count": count,
            "pct_of_baskets": round(count / total_baskets * 100, 2)
            if total_baskets
            else np.nan,
        }
        for product, count in product_counts.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def _datetime_from_mixed_formats(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _append_issue(
    issues: List[Dict[str, Any]],
    dataset: str,
    field: str,
    issue: str,
    mask: Iterable[bool],
) -> None:
    count = int(pd.Series(mask).fillna(False).sum())
    if count > 0:
        issues.append(
            {
                "dataset": dataset,
                "field": field,
                "issue": issue,
                "count": count,
            }
        )


def flag_suspicious_ranges(
    customer_info: pd.DataFrame, parsed_basket: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Flag values that fall outside basic expected ranges."""
    issues: List[Dict[str, Any]] = []
    current_year = pd.Timestamp.today().year

    _append_issue(
        issues,
        "customer_info",
        "customer_id",
        "non-positive customer_id",
        customer_info["customer_id"] <= 0,
    )

    for field in ["kids_home", "teens_home", "number_complaints", "distinct_stores_visited"]:
        if field in customer_info:
            _append_issue(
                issues,
                "customer_info",
                field,
                "negative value",
                customer_info[field] < 0,
            )

    spend_columns = [
        column for column in customer_info.columns if column.startswith("lifetime_spend_")
    ]
    for field in spend_columns + ["lifetime_total_distinct_products"]:
        if field in customer_info:
            _append_issue(
                issues,
                "customer_info",
                field,
                "negative value",
                customer_info[field] < 0,
            )

    if "percentage_of_products_bought_promotion" in customer_info:
        promo = customer_info["percentage_of_products_bought_promotion"]
        _append_issue(
            issues,
            "customer_info",
            "percentage_of_products_bought_promotion",
            "outside [0, 1]",
            promo.notna() & ~promo.between(0, 1),
        )

    if "typical_hour" in customer_info:
        hour = customer_info["typical_hour"]
        _append_issue(
            issues,
            "customer_info",
            "typical_hour",
            "outside [0, 23]",
            hour.notna() & ~hour.between(0, 23),
        )

    if "year_first_transaction" in customer_info:
        year = customer_info["year_first_transaction"]
        _append_issue(
            issues,
            "customer_info",
            "year_first_transaction",
            "outside [1900, current_year]",
            year.notna() & ~year.between(1900, current_year),
        )

    if "latitude" in customer_info:
        latitude = customer_info["latitude"]
        _append_issue(
            issues,
            "customer_info",
            "latitude",
            "outside [-90, 90]",
            latitude.notna() & ~latitude.between(-90, 90),
        )

    if "longitude" in customer_info:
        longitude = customer_info["longitude"]
        _append_issue(
            issues,
            "customer_info",
            "longitude",
            "outside [-180, 180]",
            longitude.notna() & ~longitude.between(-180, 180),
        )

    if "customer_birthdate" in customer_info:
        birthdate_raw = customer_info["customer_birthdate"]
        birthdate = _datetime_from_mixed_formats(birthdate_raw)
        age = (pd.Timestamp.today().normalize() - birthdate).dt.days / 365.25
        _append_issue(
            issues,
            "customer_info",
            "customer_birthdate",
            "unparseable non-missing birthdate",
            birthdate_raw.notna() & birthdate.isna(),
        )
        _append_issue(
            issues,
            "customer_info",
            "customer_birthdate",
            "age outside [0, 110]",
            birthdate.notna() & ~age.between(0, 110),
        )

    if parsed_basket is not None:
        _append_issue(
            issues,
            "customer_basket",
            "invoice_id",
            "non-positive invoice_id",
            parsed_basket["invoice_id"] <= 0,
        )
        _append_issue(
            issues,
            "customer_basket",
            "customer_id",
            "non-positive customer_id",
            parsed_basket["customer_id"] <= 0,
        )
        if "basket_length" in parsed_basket:
            _append_issue(
                issues,
                "customer_basket",
                "basket_length",
                "empty basket",
                parsed_basket["basket_length"] <= 0,
            )

    return pd.DataFrame(issues, columns=["dataset", "field", "issue", "count"])
