import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the metrics summary data
df = pd.read_csv("../../model_metrics_summary.csv")

# Set the style for the plots
plt.style.use("seaborn-v0_8-whitegrid")
# Use a darker color palette
sns.set_palette("dark")


# Function to plot R² for selected models for all cities on a single graph
def plot_r2_metrics():
    selected_models = ["Random Forest", "Gradient Boosting", "XGBoost", "LSTM"]
    cities = df["City"].unique()
    fig, ax = plt.subplots(figsize=(15, 8))
    x = range(len(selected_models))
    width = 0.2
    for i, city in enumerate(cities):
        city_data = df[(df["City"] == city) & (df["Model"].isin(selected_models))]
        ax.bar(
            [j + i * width for j in x],
            city_data["R2 (Test)"],
            width,
            label=f"{city}",
            alpha=0.7,
        )
    ax.set_xlabel("Models")
    ax.set_ylabel("R²")
    ax.set_title("R² for Selected Models Across Cities")
    ax.set_xticks([i + width * 2 for i in x])
    ax.set_xticklabels(selected_models, rotation=0, ha="center")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../notebooks/results/plots/selected_models_r2_all_cities.png")
    plt.close()


# Create the plots
plot_r2_metrics()
