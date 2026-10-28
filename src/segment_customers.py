"""Build RFM features, select K using two diagnostics, fit K-Means, and report personas."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "data" / "customer_segments.csv"
FIG = ROOT / "figures"
REPORT = ROOT / "reports" / "persona_summary.md"

def build_rfm(transactions):
    d = transactions.copy()
    d["order_date"] = pd.to_datetime(d["order_date"])
    as_of = d["order_date"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("customer_id").agg(
        last_purchase=("order_date", "max"), frequency=("order_id", "nunique"), monetary=("sales", "sum")
    )
    rfm["recency"] = (as_of - rfm.pop("last_purchase")).dt.days
    return rfm[["recency", "frequency", "monetary"]], as_of

def persona_name(row, medians):
    recent = row.recency <= medians.recency
    frequent = row.frequency >= medians.frequency
    high_value = row.monetary >= medians.monetary
    if recent and frequent and high_value: return "Champions"
    if recent and frequent: return "Loyal Regulars"
    if not recent and high_value: return "At-Risk High Value"
    if recent and not frequent: return "New & Promising"
    return "Occasional Customers"

def run():
    FIG.mkdir(parents=True, exist_ok=True); REPORT.parent.mkdir(parents=True, exist_ok=True)
    tx = pd.read_csv(DATA)
    rfm, as_of = build_rfm(tx)
    model_features = rfm.copy()
    model_features["recency"] = __import__("numpy").log1p(model_features["recency"])
    model_features["frequency"] = __import__("numpy").log1p(model_features["frequency"])
    model_features["monetary"] = __import__("numpy").log1p(model_features["monetary"])
    scaler = StandardScaler()
    X = scaler.fit_transform(model_features)
    scores, inertias, models = {}, {}, {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        inertias[k] = km.inertia_
        scores[k] = silhouette_score(X, labels)
        models[k] = km
    best_k = max(scores, key=scores.get)
    model = models[best_k]
    rfm["cluster_id"] = model.labels_
    medians = rfm[["recency", "frequency", "monetary"]].median()
    # Give each fitted cluster one interpretable label, using its median RFM
    # profile. This keeps the persona count aligned with the chosen k.
    cluster_profiles = rfm.groupby("cluster_id")[["recency", "frequency", "monetary"]].median()
    persona_by_cluster = {}
    used_names = set()
    for cluster_id, profile in cluster_profiles.iterrows():
        name = persona_name(profile, medians)
        if name in used_names:
            name = f"{name} (Segment {cluster_id + 1})"
        persona_by_cluster[cluster_id] = name
        used_names.add(name)
    rfm["persona"] = rfm["cluster_id"].map(persona_by_cluster)
    rfm.reset_index().to_csv(OUT, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(list(inertias), list(inertias.values()), marker="o", color="#2563eb", label="Inertia (WCSS)")
    ax.set(title="Elbow analysis", xlabel="Number of clusters (k)", ylabel="Within-cluster sum of squares")
    ax2 = ax.twinx(); ax2.plot(list(scores), list(scores.values()), marker="s", color="#16a085", label="Silhouette")
    ax2.set_ylabel("Mean silhouette coefficient", color="#12836f")
    ax.axvline(best_k, color="#64748b", linestyle="--", alpha=.7); ax.text(best_k+.08, ax.get_ylim()[1]*.93, f"selected k={best_k}")
    fig.tight_layout(); fig.savefig(FIG/"cluster_selection.png", dpi=180); plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": rfm.cluster_id.astype(str).values, "Persona": rfm.persona.values})
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.get_cmap("tab10")
    for i, (persona, group) in enumerate(plot.groupby("Persona")):
        ax.scatter(group.PC1, group.PC2, label=persona, s=45, alpha=.82, color=palette(i % 10), edgecolors="white", linewidths=.35)
    ax.legend(title="Persona", frameon=True)
    ax.set_title(f"Customer segments in PCA space (k={best_k})")
    fig.tight_layout(); fig.savefig(FIG/"pca_segments.png", dpi=180); plt.close(fig)

    summary = rfm.groupby("persona").agg(customers=("recency","size"), median_recency_days=("recency","median"), median_orders=("frequency","median"), median_spend=("monetary","median")).sort_values("customers", ascending=False)
    lines = ["# Synthetic customer persona summary", "", f"Reference date: **{as_of.date()}**. Selected **k={best_k}** by highest Silhouette among k=2..7 (score {scores[best_k]:.3f}); the Elbow curve is included as a secondary diagnostic.", "", "| Persona | Customers | Median recency (days) | Median orders | Median spend |", "|---|---:|---:|---:|---:|"]
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {int(row.customers)} | {row.median_recency_days:.0f} | {row.median_orders:.0f} | ${row.median_spend:,.2f} |")
    lines += ["", "## Suggested actions (hypotheses for testing)", "", "- **Champions:** recognize frequent, recent, higher-spend behavior with early access or loyalty benefits; monitor retention.", "- **Loyal Regulars:** test cross-category bundles and relevant replenishment reminders.", "- **New & Promising:** support a second purchase with onboarding and low-friction offers.", "- **At-Risk High Value:** test service outreach and tailored win-back messaging before broad discounts.", "- **Occasional Customers:** learn which category and purchase context predicts repeat behavior; keep engagement useful and low-frequency.", "", "Each fitted cluster receives one median-RFM-based persona label; empty labels are omitted. All source transactions are synthetic. These labels do not claim distinct real-world populations.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"customers={len(rfm)} transactions={len(tx)} selected_k={best_k} silhouette={scores[best_k]:.4f}")
    print("silhouette_by_k=" + ", ".join(f"{k}:{v:.3f}" for k,v in scores.items()))
    print(summary.to_string())
    print("explained_variance_ratio=" + str([round(x,4) for x in pca.explained_variance_ratio_]))
    return rfm, scores, inertias, best_k, pca, X

if __name__ == "__main__": run()
