"""
Visualize Model Architecture
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def create_architecture_diagram(save_path="../docs/figures/model_architecture.png"):
    """
    Create a visual diagram of the LSTM-Attention-SHAP architecture.

    Parameters:
    -----------
    save_path : str
        Path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        5,
        11.5,
        "LSTM-Attention-SHAP Architecture",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )

    # Layer definitions with positions
    layers = [
        {"name": "Input\n(30, 15)", "pos": (5, 10), "color": "#E8F4F8"},
        {
            "name": "LSTM Layer 1\n128 units\ndropout=0.2",
            "pos": (5, 8.5),
            "color": "#B8E6F0",
        },
        {
            "name": "LSTM Layer 2\n64 units\ndropout=0.2",
            "pos": (5, 7),
            "color": "#88D8E8",
        },
        {
            "name": "Attention Layer\nBahdanau-style\n64 units",
            "pos": (5, 5.5),
            "color": "#FFD88F",
        },
        {"name": "Context Vector\nWeighted Sum", "pos": (5, 4), "color": "#FFC470"},
        {
            "name": "Dense Layer\n32 units + ReLU\ndropout=0.3",
            "pos": (5, 2.5),
            "color": "#D4A5FF",
        },
    ]

    outputs = [
        {"name": "Volatility\nForecast", "pos": (3, 0.8), "color": "#90EE90"},
        {"name": "VaR 99%\nForecast", "pos": (7, 0.8), "color": "#FF9999"},
    ]

    # Draw main layers
    box_width = 2.5
    box_height = 0.8

    for layer in layers:
        box = FancyBboxPatch(
            (layer["pos"][0] - box_width / 2, layer["pos"][1] - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor=layer["color"],
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            layer["pos"][0],
            layer["pos"][1],
            layer["name"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Draw output heads
    for output in outputs:
        box = FancyBboxPatch(
            (output["pos"][0] - box_width / 2, output["pos"][1] - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor=output["color"],
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            output["pos"][0],
            output["pos"][1],
            output["name"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Draw arrows between layers
    for i in range(len(layers) - 1):
        arrow = FancyArrowPatch(
            (layers[i]["pos"][0], layers[i]["pos"][1] - box_height / 2 - 0.1),
            (layers[i + 1]["pos"][0], layers[i + 1]["pos"][1] + box_height / 2 + 0.1),
            arrowstyle="->,head_width=0.4,head_length=0.4",
            color="black",
            linewidth=2,
        )
        ax.add_patch(arrow)

    # Draw arrows from Dense to outputs
    for output in outputs:
        arrow = FancyArrowPatch(
            (layers[-1]["pos"][0], layers[-1]["pos"][1] - box_height / 2 - 0.1),
            (output["pos"][0], output["pos"][1] + box_height / 2 + 0.1),
            arrowstyle="->,head_width=0.4,head_length=0.4",
            color="black",
            linewidth=2,
        )
        ax.add_patch(arrow)

    # Add attention mechanism annotation
    ax.annotate(
        "Attention Weights\n(Dynamic Focus)",
        xy=(5, 5.5),
        xytext=(8.5, 5.5),
        fontsize=9,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="orange"),
    )

    # Add SHAP annotation
    ax.annotate(
        "SHAP Layer\n(Explainability)",
        xy=(5, 2.5),
        xytext=(1.5, 2.5),
        fontsize=9,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="blue"),
    )

    # Add specifications box
    spec_text = (
        "Model Specifications:\n"
        "• Total Parameters: ~147,000\n"
        "• Optimizer: Adam (lr=1e-3)\n"
        "• Batch Size: 64\n"
        "• Lookback Window: 30 days\n"
        "• Features: 15 dimensions"
    )
    ax.text(
        0.3,
        6.5,
        spec_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
    )

    # Add loss functions box
    loss_text = (
        "Loss Functions:\n"
        "• Volatility: MSE\n"
        "• VaR: Pinball Loss (τ=0.01)\n"
        "• Combined with weights\n"
        "  [1.0, 0.5]"
    )
    ax.text(
        9.7,
        6.5,
        loss_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcoral", alpha=0.3),
        verticalalignment="top",
        ha="right",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Architecture diagram saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    create_architecture_diagram()
    print("Architecture visualization complete!")
