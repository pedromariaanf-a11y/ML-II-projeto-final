"""Data loading utilities for the customer segmentation project."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


CUSTOMER_INFO_FILE = "customer_info.csv"
CUSTOMER_BASKET_FILE = "customer_basket.csv"
PROJECT_FILES_DIR = "Project files"


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


def resolve_data_file(filename: str, root: Optional[str | Path] = None) -> Path:
    """Resolve a raw data file from the current organized project layout."""
    base = Path(root) if root is not None else project_root()
    candidates = [
        base / filename,
        base / PROJECT_FILES_DIR / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {attempted}")


def load_customer_info(root: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the customer-level dataset."""
    path = resolve_data_file(CUSTOMER_INFO_FILE, root)
    return pd.read_csv(path)


def load_customer_basket(root: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the sampled basket dataset."""
    path = resolve_data_file(CUSTOMER_BASKET_FILE, root)
    return pd.read_csv(path)


def load_datasets(root: Optional[str | Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw project datasets."""
    return load_customer_info(root), load_customer_basket(root)
