import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_kmeans(df):
    """
    Run K-Means clustering on customer data.
    Returns cluster labels and saves a scatter plot.
    """
    # Aggregate per customer: total amount and frequency
    customer_df = df.groupby('customer_name').agg(
        total_amount=('amount', 'sum'),
        frequency=('amount', 'count')
    ).reset_index()

    features = customer_df[['total_amount', 'frequency']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df['cluster'] = kmeans.fit_predict(scaled)

    # Calculate clustering quality metrics
    silhouette = silhouette_score(scaled, customer_df['cluster'])
    davies_bouldin = davies_bouldin_score(scaled, customer_df['cluster'])
    calinski = calinski_harabasz_score(scaled, customer_df['cluster'])
    inertia = kmeans.inertia_

    # Print metrics to console
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING EVALUATION METRICS")
    print("="*60)
    print(f"Silhouette Score:        {silhouette:.4f}  (Range: -1 to 1, Higher is better)")
    print(f"Davies-Bouldin Index:    {davies_bouldin:.4f}  (Lower is better)")
    print(f"Calinski-Harabasz Score: {calinski:.2f}  (Higher is better)")
    print(f"Inertia (Within-cluster): {inertia:.2f}  (Lower is better)")
    print("="*60)
    print("\nInterpretation:")
    if silhouette > 0.5:
        print("✓ Excellent clustering - clusters are well separated")
    elif silhouette > 0.3:
        print("✓ Good clustering - reasonable cluster separation")
    else:
        print("⚠ Fair clustering - clusters may overlap")
    print("="*60 + "\n")

    # Map clusters to labels based on mean total_amount
    cluster_means = customer_df.groupby('cluster')['total_amount'].mean().sort_values()
    label_map = {
        cluster_means.index[0]: 'Low Value',
        cluster_means.index[1]: 'Medium Value',
        cluster_means.index[2]: 'High Value'
    }
    customer_df['segment'] = customer_df['cluster'].map(label_map)

    # Save scatter plot
    colors = {'High Value': '#e74c3c', 'Medium Value': '#f39c12', 'Low Value': '#2ecc71'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for seg, grp in customer_df.groupby('segment'):
        ax.scatter(grp['frequency'], grp['total_amount'],
                   label=seg, color=colors[seg], alpha=0.8, s=80)
    ax.set_xlabel('Purchase Frequency')
    ax.set_ylabel('Total Purchase Amount')
    ax.set_title(f'Customer Segmentation (K-Means) | Silhouette Score: {silhouette:.3f}')
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return customer_df[['customer_name', 'total_amount', 'frequency', 'segment']].to_dict(orient='records')
