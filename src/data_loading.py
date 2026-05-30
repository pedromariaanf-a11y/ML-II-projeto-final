"""Data loading utilities for the customer segmentation project."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


CUSTOMER_INFO_FILE = "customer_info.csv"
CUSTOMER_BASKET_FILE = "customer_basket.csv"


def project_root() -> Path:
    """Return the repository root inferred from this module location."""
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path, root: Optional[str | Path] = None) -> Path:
    """Resolve a project-relative path without hardcoding local absolute paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    base = Path(root) if root is not None else project_root()
    return base / candidate


def load_customer_info(root: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the customer-level dataset."""
    path = resolve_repo_path(CUSTOMER_INFO_FILE, root)
    return pd.read_csv(path)


def load_customer_basket(root: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the sampled basket dataset."""
    path = resolve_repo_path(CUSTOMER_BASKET_FILE, root)
    return pd.read_csv(path)


def load_datasets(root: Optional[str | Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw project datasets."""
    return load_customer_info(root), load_customer_basket(root)
