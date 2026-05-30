# AGENTS.md

## Project Objective

Create a reproducible customer segmentation project with clean code, interpretable clusters, targeted campaign recommendations, and a final CSV containing every `customer_id` and its assigned cluster.

## Permanent Instructions

- Preserve raw CSV files unchanged.
- Do not create final clustering outputs until a clustering workflow exists and has been reviewed.
- Keep reusable logic in `src/`; keep notebooks focused on analysis narrative and outputs.
- Use relative repository paths only; do not hardcode local absolute paths.
- Ensure all customers in `customer_info.csv` are considered in the final clustering deliverable.
- Treat basket-derived features as optional inputs because some customers do not appear in `customer_basket.csv`.
- Update `PROJECT_STATE.md` after meaningful project milestones.

## Coding Conventions

- Prefer small, testable functions.
- Use deterministic random seeds when modeling is introduced.
- Separate data loading, audit/preprocessing, modeling, and reporting concerns.
- Create `DECISIONS.md` or `EXPERIMENT_LOG.md` only if they add clear value.
