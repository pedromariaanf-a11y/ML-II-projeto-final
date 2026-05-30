"""Simple audit helpers used by the first notebook."""

import ast
from collections import Counter

import numpy as np
import pandas as pd


def dataframe_overview(df):
    """Show column names, data types, and missing-value counts."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().to_numpy(),
            "missing_count": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean().to_numpy() * 100).round(2),
        }
    )


def missing_values(df):
    """Show only the columns that have missing values."""
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


def duplicate_summary(customer_info, customer_basket):
    """Check duplicate rows and duplicate IDs in both datasets."""
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


def validate_customer_overlap(customer_info, customer_basket, sample_size=10):
    """Check whether basket customers exist in the full customer table."""
    info_ids = set(customer_info["customer_id"].dropna())
    basket_ids = set(customer_basket["customer_id"].dropna())

    basket_ids_missing_in_info = sorted(basket_ids - info_ids)
    customers_without_baskets = sorted(info_ids - basket_ids)

    if info_ids:
        customers_without_baskets_pct = round(
            len(customers_without_baskets) / len(info_ids) * 100, 2
        )
    else:
        customers_without_baskets_pct = np.nan

    return {
        "customer_info_unique_customers": len(info_ids),
        "customer_basket_unique_customers": len(basket_ids),
        "basket_ids_missing_in_info": len(basket_ids_missing_in_info),
        "sample_basket_ids_missing_in_info": basket_ids_missing_in_info[:sample_size],
        "customers_without_baskets": len(customers_without_baskets),
        "customers_without_baskets_pct": customers_without_baskets_pct,
        "sample_customers_without_baskets": customers_without_baskets[:sample_size],
    }


def parse_goods_column(customer_basket):
    """Parse the product-list strings and calculate basket length."""
    parsed_goods = []
    errors = []

    for row_index, value in customer_basket["list_of_goods"].items():
        try:
            goods = parse_one_goods_list(value)
            parsed_goods.append(goods)
        except (SyntaxError, ValueError, TypeError) as error:
            parsed_goods.append([])
            errors.append(
                {
                    "row_index": row_index,
                    "raw_value": value,
                    "error": str(error),
                }
            )

    parsed_basket = customer_basket.copy()
    parsed_basket["goods"] = parsed_goods
    parsed_basket["basket_length"] = [len(goods) for goods in parsed_goods]

    return parsed_basket, pd.DataFrame(errors)


def basket_length_distribution(parsed_basket):
    """Summarize how many products appear in each basket."""
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return (
        parsed_basket["basket_length"]
        .describe(percentiles=percentiles)
        .rename("basket_length")
        .reset_index()
        .rename(columns={"index": "metric"})
    )


def top_products(parsed_basket, top_n=20):
    """Count the most common products across all sampled baskets."""
    product_counts = Counter()
    for goods in parsed_basket["goods"]:
        product_counts.update(goods)

    total_baskets = len(parsed_basket)
    rows = []
    for product, count in product_counts.most_common(top_n):
        rows.append(
            {
                "product": product,
                "count": count,
                "pct_of_baskets": round(count / total_baskets * 100, 2)
                if total_baskets
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def flag_suspicious_ranges(customer_info, parsed_basket=None):
    """Flag basic values that fall outside expected ranges."""
    issues = []
    current_year = pd.Timestamp.today().year

    add_issue(
        issues,
        "customer_info",
        "customer_id",
        "non-positive customer_id",
        customer_info["customer_id"] <= 0,
    )

    for field in ["kids_home", "teens_home", "number_complaints", "distinct_stores_visited"]:
        if field in customer_info:
            add_issue(issues, "customer_info", field, "negative value", customer_info[field] < 0)

    spend_columns = [
        column for column in customer_info.columns if column.startswith("lifetime_spend_")
    ]
    for field in spend_columns + ["lifetime_total_distinct_products"]:
        if field in customer_info:
            add_issue(issues, "customer_info", field, "negative value", customer_info[field] < 0)

    if "percentage_of_products_bought_promotion" in customer_info:
        promotion_pct = customer_info["percentage_of_products_bought_promotion"]
        add_issue(
            issues,
            "customer_info",
            "percentage_of_products_bought_promotion",
            "outside [0, 1]",
            promotion_pct.notna() & ~promotion_pct.between(0, 1),
        )

    if "typical_hour" in customer_info:
        typical_hour = customer_info["typical_hour"]
        add_issue(
            issues,
            "customer_info",
            "typical_hour",
            "outside [0, 23]",
            typical_hour.notna() & ~typical_hour.between(0, 23),
        )

    if "year_first_transaction" in customer_info:
        first_year = customer_info["year_first_transaction"]
        add_issue(
            issues,
            "customer_info",
            "year_first_transaction",
            "outside [1900, current_year]",
            first_year.notna() & ~first_year.between(1900, current_year),
        )

    if "latitude" in customer_info:
        latitude = customer_info["latitude"]
        add_issue(
            issues,
            "customer_info",
            "latitude",
            "outside [-90, 90]",
            latitude.notna() & ~latitude.between(-90, 90),
        )

    if "longitude" in customer_info:
        longitude = customer_info["longitude"]
        add_issue(
            issues,
            "customer_info",
            "longitude",
            "outside [-180, 180]",
            longitude.notna() & ~longitude.between(-180, 180),
        )

    if "customer_birthdate" in customer_info:
        birthdate_raw = customer_info["customer_birthdate"]
        birthdate = parse_dates(birthdate_raw)
        age = (pd.Timestamp.today().normalize() - birthdate).dt.days / 365.25
        add_issue(
            issues,
            "customer_info",
            "customer_birthdate",
            "unparseable non-missing birthdate",
            birthdate_raw.notna() & birthdate.isna(),
        )
        add_issue(
            issues,
            "customer_info",
            "customer_birthdate",
            "age outside [0, 110]",
            birthdate.notna() & ~age.between(0, 110),
        )

    if parsed_basket is not None:
        add_issue(
            issues,
            "customer_basket",
            "invoice_id",
            "non-positive invoice_id",
            parsed_basket["invoice_id"] <= 0,
        )
        add_issue(
            issues,
            "customer_basket",
            "customer_id",
            "non-positive customer_id",
            parsed_basket["customer_id"] <= 0,
        )
        if "basket_length" in parsed_basket:
            add_issue(
                issues,
                "customer_basket",
                "basket_length",
                "empty basket",
                parsed_basket["basket_length"] <= 0,
            )

    return pd.DataFrame(issues, columns=["dataset", "field", "issue", "count"])


def parse_one_goods_list(value):
    """Parse one `list_of_goods` value."""
    parsed = value if isinstance(value, list) else ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError("list_of_goods value is not a list")
    return [str(item) for item in parsed]


def parse_dates(values):
    """Parse mixed date strings while staying compatible with older pandas."""
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def add_issue(issues, dataset, field, issue, mask):
    """Append an issue row only when at least one row is affected."""
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
