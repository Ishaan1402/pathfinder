import os
import sys

# Ensure pandas and matplotlib are installed
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Required packages (pandas, matplotlib) are missing.")
    print("Installing packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib"])
    import pandas as pd
    import matplotlib.pyplot as plt

def plot_study_results():
    csv_file = "bridge_crack_study_500px.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run extract_to_csv.py first.")
        return

    # Load data
    df = pd.read_csv(csv_file)
    print("Loaded data:")
    print(df[["trial_number", "score", "loss", "learning_rate", "batch_size", "resolution"]])

    if len(df) == 0:
        print("No completed trials to plot.")
        return

    # Modern styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color definitions
    accent_color = "#3B82F6"  # Premium Indigo/Blue
    highlight_color = "#10B981"  # Emerald Green for best
    neutral_dark = "#1F2937"

    # Plot 1: Score Progression over completed trials
    ax1.plot(df["trial_number"], df["score"], marker="o", color=accent_color, linewidth=2, markersize=8, label="Dice Score")
    
    # Highlight best trial
    best_idx = df["score"].idxmax()
    best_trial = df.loc[best_idx]
    ax1.scatter(best_trial["trial_number"], best_trial["score"], color=highlight_color, s=200, zorder=5, label=f"Best (Trial #{int(best_trial['trial_number'])}: {best_trial['score']:.4f})")
    
    # Labels and Titles
    ax1.set_title("Dice Score Progression (Resolution >= 500px)", fontsize=14, fontweight="bold", pad=15, color=neutral_dark)
    ax1.set_xlabel("Optuna Trial Number", fontsize=12, labelpad=10)
    ax1.set_ylabel("Dice Score (Higher is Better)", fontsize=12, labelpad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E5E7EB")
    ax1.set_ylim(df["score"].min() - 0.005, df["score"].max() + 0.005)

    # Plot 2: Learning Rate vs Score colored by Batch Size
    scatter = ax2.scatter(
        df["learning_rate"], 
        df["score"], 
        c=df["batch_size"], 
        cmap="viridis", 
        s=120, 
        edgecolors="none",
        alpha=0.85
    )
    # Highlight best trial in scatter
    ax2.scatter(
        best_trial["learning_rate"], 
        best_trial["score"], 
        color=highlight_color, 
        edgecolors="black", 
        s=250, 
        zorder=5, 
        label="Best Model"
    )

    ax2.set_xscale("log")
    ax2.set_title("Learning Rate vs. Score (Size: Batch Size)", fontsize=14, fontweight="bold", pad=15, color=neutral_dark)
    ax2.set_xlabel("Learning Rate (Log Scale)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Dice Score", fontsize=12, labelpad=10)
    
    # Colorbar for Batch Size
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Batch Size", fontsize=11, rotation=270, labelpad=15)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#E5E7EB")

    plt.tight_layout()
    plot_path = "bridge_crack_500px_plots.png"
    plt.savefig(plot_path, dpi=300, facecolor="white")
    print(f"Successfully generated and saved plots to '{plot_path}'.")
    plt.close()

if __name__ == "__main__":
    plot_study_results()
