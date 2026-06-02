"""Small modeling helpers for the clustering notebooks."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def apply_modeling_transforms(df, log_columns=None):
    """Apply simple transformations before scaling."""
    transformed = df.copy()
    log_columns = log_columns or []

    for column in log_columns:
        if column in transformed.columns:
            transformed[f"log_{column}"] = np.log1p(transformed[column])
            transformed = transformed.drop(columns=[column])

    return transformed


def prepare_modeling_matrix(feature_table, feature_columns, log_columns=None):
    """Create a transformed and scaled modeling matrix."""
    modeling_data = feature_table[feature_columns].copy()
    transformed_data = apply_modeling_transforms(modeling_data, log_columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(transformed_data)
    return transformed_data, X_scaled, scaler


def fit_kmeans_labels(X_scaled, k, random_state=42):
    """Fit K-Means and return cluster labels."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return model.fit_predict(X_scaled)


def add_cluster_labels(df, labels, cluster_column="cluster"):
    """Add cluster labels to a dataframe copy without saving them to disk."""
    result = df.copy()
    result[cluster_column] = labels
    return result


def summarize_cluster_sizes(labels):
    """Count customers in each cluster."""
    counts = pd.Series(labels).value_counts().sort_index()
    summary = counts.rename_axis("cluster").reset_index(name="cluster_size")
    summary["cluster_percentage"] = summary["cluster_size"] / len(labels)
    return summary


def profile_clusters(df, cluster_column, profile_columns):
    """Calculate mean profile values for each cluster."""
    existing_columns = [column for column in profile_columns if column in df.columns]
    profile = df.groupby(cluster_column)[existing_columns].mean().reset_index()
    counts = df.groupby(cluster_column).size().rename("customer_count").reset_index()
    return counts.merge(profile, on=cluster_column, how="left")


def compare_cluster_profiles_to_global(df, cluster_column, profile_columns):
    """Compare each cluster mean with the global feature average."""
    profile = profile_clusters(df, cluster_column, profile_columns)
    existing_columns = [column for column in profile_columns if column in df.columns]
    global_average = df[existing_columns].mean()

    comparison = profile.copy()
    for column in existing_columns:
        comparison[column] = comparison[column] - global_average[column]

    return comparison
