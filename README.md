# Customer Segmentation Study — CodeNova Task 4

Reproducible customer segmentation from RFM features (Recency, Frequency, Monetary value), with scaling, empirical Elbow and Silhouette analysis, K-Means clustering, PCA projections, and actionable persona summaries.

> **Synthetic-data disclosure:** the assignment allows a public or synthetic dataset. This repository uses fictional transaction records generated with a fixed seed. No real people, customer IDs, or business records are represented.

## Deliverables

- `notebooks/customer_segmentation_study.ipynb` — clustering pipeline, metrics, persona interpretation, and plots.
- `data/transactions.csv` — deterministic synthetic transaction sample.
- `data/customer_segments.csv` — customer-level RFM measurements, cluster IDs, and readable persona labels.
- `figures/` — Elbow/Silhouette and PCA segment plots.
- `reports/persona_summary.md` — cluster sizes and behavioral summaries with suggested actions.
- `src/generate_transactions.py`, `src/segment_customers.py` — reproducible generation and analysis scripts.

## Run locally

Python 3.10+:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/generate_transactions.py
python src/segment_customers.py
jupyter notebook notebooks/customer_segmentation_study.ipynb
```

## Method and mathematical choices

RFM is computed as of the day after the final transaction date. Recency is days since the latest purchase; frequency is distinct orders; monetary is total spend. Because frequency and spend are right-skewed, the model applies `log1p` to all three measures and standardizes with `StandardScaler`. K-Means uses a fixed seed and multiple initializations. Candidate k values are compared using within-cluster sum of squares (Elbow) and mean Silhouette coefficient; the chosen k maximizes the evaluated Silhouette score. PCA is a two-dimensional visualization only; clustering is fitted in the full scaled RFM space.

## Limitations and responsible use

This controlled synthetic example is for learning, not targeting real people. The generator intentionally creates varied purchasing behavior, so these clusters are not evidence of market structure. K-Means assumes roughly compact Euclidean groups, is sensitive to scale and initialization, and assigns each record to a cluster. Silhouette and Elbow are diagnostics, not proof that segments are true or useful. Persona names summarize this generated sample; validate segment stability, business utility, fairness, privacy, and consent before any real-world use.
