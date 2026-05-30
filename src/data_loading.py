"""Simple data loading helpers for the project notebooks."""

from pathlib import Path

import pandas as pd


CUSTOMER_INFO_FILE = "customer_info.csv"
CUSTOMER_BASKET_FILE = "customer_basket.csv"
PROJECT_FILES_DIR = "Project files"


def project_root():
    """Find the repository root from the location of this file."""
    return Path(__file__).resolve().parents[1]


def resolve_data_file(filename, root=None):
    """Find a raw data file in the current project layout."""
    base = Path(root) if root is not None else project_root()

    possible_paths = [
        base / filename,
        base / PROJECT_FILES_DIR / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    checked_paths = ", ".join(str(path) for path in possible_paths)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {checked_paths}")


def load_customer_info(root=None):
    """Load the customer-level dataset."""
    return pd.read_csv(resolve_data_file(CUSTOMER_INFO_FILE, root))


def load_customer_basket(root=None):
    """Load the sampled basket dataset."""
    return pd.read_csv(resolve_data_file(CUSTOMER_BASKET_FILE, root))


def load_datasets(root=None):
    """Load both raw datasets."""
    customer_info = load_customer_info(root)
    customer_basket = load_customer_basket(root)
    return customer_info, customer_basket
