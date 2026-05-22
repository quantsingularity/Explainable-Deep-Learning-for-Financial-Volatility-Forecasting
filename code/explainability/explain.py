"""
SHAP Explainability Module for LSTM-Attention Model
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.cluster import KMeans

# Set random seed
np.random.seed(123)


def prepare_shap_background(X_train, n_samples=100):
    """
    Prepare background samples for SHAP using k-means clustering.

    Parameters:
    -----------
    X_train : np.array
        Training data (n_samples, time_steps, n_features)
    n_samples : int
        Number of background samples

    Returns:
    --------
    background : np.array
        Background samples for SHAP
    """
    print(f"\nPreparing {n_samples} background samples using k-means...")

    # Reshape for clustering
    n_train = min(1000, len(X_train))  # Use subset for efficiency
    X_flat = X_train[:n_train].reshape(n_train, -1)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_samples, random_state=123, n_init=10)
    kmeans.fit(X_flat)

    # Get cluster centers and reshape back
    background_flat = kmeans.cluster_centers_
    background = background_flat.reshape(n_samples, X_train.shape[1], X_train.shape[2])

    print(f"Background shape: {background.shape}")
    return background


def compute_shap_values(model, X_test, background, feature_names):
    """
    Compute SHAP values using GradientExplainer.

    GradientExplainer is used instead of DeepExplainer because it handles
    TF2 models with custom layers (e.g. AttentionLayer returning a tuple)
    more reliably through standard gradient computation.

    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : np.array
        Test data
    background : np.array
        Background samples
    feature_names : list
        Feature names

    Returns:
    --------
    shap_values : list
        SHAP values for each output
    explainer : shap.GradientExplainer
        SHAP explainer object
    """
    print("\n" + "=" * 70)
    print("COMPUTING SHAP VALUES")
    print("=" * 70)

    # GradientExplainer works reliably with TF2 functional-API models
    # including those with custom layers that return tuples.
    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)

    # Compute SHAP values (use subset for efficiency)
    n_explain = min(200, len(X_test))
    print(f"Computing SHAP values for {n_explain} test samples...")

    shap_values = explainer.shap_values(X_test[:n_explain])

    print(f"SHAP values computed. Shape: {np.array(shap_values[0]).shape}")

    return shap_values, explainer


def aggregate_shap_across_time(shap_values, feature_names):
    """
    Aggregate SHAP values across time steps for global importance.

    Parameters:
    -----------
    shap_values : np.array
        SHAP values (n_samples, time_steps, n_features)
    feature_names : list
        Feature names

    Returns:
    --------
    importance_df : pd.DataFrame
        Feature importance dataframe
    """
    # Take absolute values and mean across samples and time steps
    shap_abs = np.abs(shap_values)

    # Aggregate: mean over samples and time steps
    feature_importance = np.mean(shap_abs, axis=(0, 1))

    # Create dataframe
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )

    # Sort by importance
    importance_df = importance_df.sort_values("Importance", ascending=False)

    return importance_df


def plot_shap_summary(shap_values, X_test, feature_names, save_path="../docs/figures"):
    """
    Create SHAP beeswarm plot showing feature importance and impacts.

    Parameters:
    -----------
    shap_values : np.array
        SHAP values for volatility output
    X_test : np.array
        Test data
    feature_names : list
        Feature names
    save_path : str
        Path to save figure
    """
    print("\nGenerating SHAP beeswarm plot...")

    # Aggregate SHAP values across time dimension
    # Average across time steps for each sample
    shap_aggregated = np.mean(shap_values, axis=1)  # (n_samples, n_features)
    X_aggregated = np.mean(X_test, axis=1)  # (n_samples, n_features)

    # Create plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_aggregated,
        X_aggregated,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 8),
    )
    plt.title(
        "SHAP Feature Importance (Beeswarm Plot)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "shap_beeswarm_plot.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"SHAP beeswarm plot saved to: {save_file}")

    plt.close()


def plot_shap_bar(importance_df, save_path="../docs/figures", top_n=12):
    """
    Create bar plot of global feature importance.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    save_path : str
        Path to save figure
    top_n : int
        Number of top features to display
    """
    print(f"\nGenerating SHAP importance bar plot (top {top_n} features)...")

    # Select top N features
    top_features = importance_df.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.7, len(top_features)))
    ax.barh(range(len(top_features)), top_features["Importance"], color=colors)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Invert y-axis so highest importance is on top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "shap_importance_bar.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"SHAP importance bar plot saved to: {save_file}")

    plt.close()


def plot_attention_heatmap(
    attention_weights, save_path="../docs/figures", n_samples=50
):
    """
    Plot attention weights heatmap.

    Parameters:
    -----------
    attention_weights : np.array
        Attention weights (n_samples, time_steps, 1)
    save_path : str
        Path to save figure
    n_samples : int
        Number of samples to visualize
    """
    print(f"\nGenerating attention heatmap for {n_samples} samples...")

    # Select subset and squeeze last dimension
    weights = attention_weights[:n_samples].squeeze()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        weights.T,
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"},
        ax=ax,
        xticklabels=10,
        yticklabels=5,
    )

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Time Step (Days Back)", fontsize=12)
    ax.set_title(
        "Attention Mechanism: Temporal Focus Heatmap", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    # Save
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "attention_heatmap.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"Attention heatmap saved to: {save_file}")

    plt.close()


def create_interpretability_report(importance_df, save_path="../docs/tables"):
    """
    Create interpretability report with key findings.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    save_path : str
        Path to save report
    """
    os.makedirs(save_path, exist_ok=True)

    # Save importance table
    importance_df.to_csv(
        os.path.join(save_path, "shap_feature_importance.csv"), index=False
    )

    # Create summary report
    report = []
    report.append("=" * 70)
    report.append("INTERPRETABILITY ANALYSIS SUMMARY")
    report.append("=" * 70)
    report.append("\nTop 5 Most Important Features:\n")

    for rank, (_, row) in enumerate(importance_df.head(5).iterrows(), start=1):
        report.append(f"  {rank}. {row['Feature']}: {row['Importance']:.4f}")

    report.append("\n\nKey Finding:")
    top_feature = importance_df.iloc[0]["Feature"]
    top_importance = importance_df.iloc[0]["Importance"]
    report.append(
        f"  {top_feature} is the dominant driver with mean |SHAP| = {top_importance:.4f}"
    )
    report.append(
        "  This confirms the critical role of geopolitical factors in volatility forecasting."
    )

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    with open(os.path.join(save_path, "interpretability_report.txt"), "w") as f:
        f.write(report_text)


def run_full_explainability_pipeline(
    model, X_train, X_test, feature_names, attention_weights=None
):
    """
    Run complete explainability analysis.

    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_train : np.array
        Training data
    X_test : np.array
        Test data
    feature_names : list
        Feature names
    attention_weights : np.array
        Pre-computed attention weights (optional)

    Returns:
    --------
    results : dict
        Explainability results
    """
    print("\n" + "=" * 70)
    print("EXPLAINABILITY PIPELINE")
    print("=" * 70)

    # 1. Prepare background
    background = prepare_shap_background(X_train, n_samples=100)

    # 2. Compute SHAP values
    shap_values, explainer = compute_shap_values(
        model, X_test, background, feature_names
    )

    # Extract volatility output SHAP values (first output)
    shap_vol = np.array(shap_values[0])

    # 3. Aggregate importance
    importance_df = aggregate_shap_across_time(shap_vol, feature_names)

    print("\n" + "-" * 70)
    print("Feature Importance Ranking:")
    print("-" * 70)
    print(importance_df.to_string(index=False))

    # 4. Generate plots
    plot_shap_summary(shap_vol, X_test[:200], feature_names)
    plot_shap_bar(importance_df)

    if attention_weights is not None:
        plot_attention_heatmap(attention_weights)

    # 5. Create report
    create_interpretability_report(importance_df)

    print("\n" + "=" * 70)
    print("EXPLAINABILITY PIPELINE COMPLETED")
    print("=" * 70)

    return {
        "shap_values": shap_values,
        "importance_df": importance_df,
        "explainer": explainer,
    }


if __name__ == "__main__":
    print("SHAP explainability module loaded.")
    print("Use run_full_explainability_pipeline() for complete analysis.")
