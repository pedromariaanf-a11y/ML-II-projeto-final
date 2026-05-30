# Customer Segmentation Project

## Objective

Build a reproducible customer segmentation solution for the Machine Learning II assignment. The final project should identify meaningful customer groups, explain their behavior, propose targeted marketing campaigns, and export a CSV with every `customer_id` assigned to a cluster.

## Dataset Description

The repository starts with two raw datasets and the original assignment brief:

- `customer_info.csv`: one row per customer with demographics, household information, complaints, loyalty-card status, lifetime spending by category, location, distinct products, and first transaction year.
- `customer_basket.csv`: 100,000 sampled shopping baskets with `invoice_id`, `customer_id`, and `list_of_goods` stored as a Python-style list string.
- `Machine Learning II - Project Statement (1).pdf`: the assignment statement.

Raw CSV files must remain unchanged. The first workflow only audits the data; it does not perform clustering or create final outputs.

## Setup Instructions

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Audit Notebook

Start Jupyter from the repository root:

```powershell
python -m notebook notebooks/01_data_audit.ipynb
```

Then run all cells in `notebooks/01_data_audit.ipynb`. The notebook loads both datasets, checks schema and data quality, validates customer ID overlap, parses basket goods, summarizes basket sizes, reports top products, and flags suspicious value ranges.

For a command-line validation run:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit.ipynb
```

## Repository Structure

```text
.
|-- AGENTS.md
|-- PROJECT_STATE.md
|-- README.md
|-- requirements.txt
|-- customer_basket.csv
|-- customer_info.csv
|-- Machine Learning II - Project Statement (1).pdf
|-- data/
|   `-- .gitkeep
|-- notebooks/
|   `-- 01_data_audit.ipynb
|-- outputs/
|   `-- .gitkeep
`-- src/
    |-- __init__.py
    |-- data_audit.py
    `-- data_loading.py
```
