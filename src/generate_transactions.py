"""Create a reproducible, entirely fictional retail transaction sample."""
from pathlib import Path
import csv, random, math
from datetime import date, timedelta

OUT = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
PROFILES = [
    ("high_value_frequent", .17, 17, 1000, 12),
    ("steady_regular", .28, 42, 380, 38),
    ("recent_low_spend", .25, 95, 150, 5),
    ("lapsed_high_value", .15, 470, 730, 14),
    ("occasional", .15, 230, 230, 8),
]

def generate(seed=42, customers=600):
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    rows, cid = [], 0
    for profile, share, recency, spend, freq in PROFILES:
        for _ in range(round(customers * share)):
            cid += 1
            customer = f"C{cid:04d}"
            base = start + timedelta(days=rng.randint(0, 365))
            last = date(2025, 12, 31) - timedelta(days=max(1, int(rng.gauss(recency, max(10, recency * .22)))))
            last = max(base, last)
            count = max(1, int(rng.gauss(freq, max(1, freq * .22))))
            if profile == "high_value_frequent": count = max(count, 8)
            span = max(1, (last - base).days)
            order_dates = sorted(base + timedelta(days=rng.randint(0, span)) for _ in range(count))
            for j, order_date in enumerate(order_dates, 1):
                basket = max(12, rng.lognormvariate(math.log(max(25, spend / max(1, freq))), .48))
                rows.append({"customer_id": customer, "order_id": f"{customer}-O{j:03d}",
                             "order_date": order_date.isoformat(), "sales": round(basket, 2),
                             "units": rng.randint(1, 6), "category": rng.choice(["Technology", "Furniture", "Office Supplies"])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["customer_id", "order_id", "order_date", "sales", "units", "category"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows):,} synthetic transactions for {cid:,} fictional customers to {OUT}")

if __name__ == "__main__": generate()
