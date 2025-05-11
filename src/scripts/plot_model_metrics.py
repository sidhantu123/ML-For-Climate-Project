import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the metrics summary data
df = pd.read_csv("../../model_metrics_summary.csv")

# Set the style for the plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")


# Function to plot RMSE and MAE metrics for each city
def plot_rmse_mae():
    cities = df["City"].unique()
    for city in cities:
        city_data = df[df["City"] == city]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(city_data))
        width = 0.35
        ax.bar(x, city_data["RMSE (Test)"], width, label="RMSE", alpha=0.7)
        ax.bar(
            [i + width for i in x],
            city_data["MAE (Test)"],
            width,
            label="MAE",
            alpha=0.7,
        )
        ax.set_xlabel("Models")
        ax.set_ylabel("Error")
        ax.set_title(f"RMSE and MAE for {city}")
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(city_data["Model"], rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"../notebooks/results/plots/{city}/{city}_rmse_mae.png")
        plt.close()


# Function to plot R² metrics for each city
def plot_r2():
    cities = df["City"].unique()
    for city in cities:
        city_data = df[df["City"] == city]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(city_data)), city_data["R2 (Test)"], alpha=0.7)
        ax.set_xlabel("Models")
        ax.set_ylabel("R²")
        ax.set_title(f"R² for {city}")
        ax.set_xticks(range(len(city_data)))
        ax.set_xticklabels(city_data["Model"], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"../notebooks/results/plots/{city}/{city}_r2.png")
        plt.close()


# Create the plots
plot_rmse_mae()
plot_r2()
