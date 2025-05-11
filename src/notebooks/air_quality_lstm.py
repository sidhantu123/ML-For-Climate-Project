# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import json
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_palette("viridis")

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# Cell 2: Define LSTM model class
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Cell 3: Define utility functions
def load_city_data(city_name):
    """Load preprocessed data for a specific city"""
    data_dir = "../data/processed"
    city_data = {}

    for split in ["train", "val", "test"]:
        file_path = f"{data_dir}/{city_name}/{split}.csv"
        if os.path.exists(file_path):
            city_data[split] = pd.read_csv(file_path)
            city_data[split]["date"] = pd.to_datetime(city_data[split]["Date"])

    return city_data


def prepare_lstm_data(city_data, sequence_length=7, target_scaler=None):
    """Prepare data for LSTM model with sequence creation and target scaling"""
    numeric_cols = city_data["train"].select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "AQI"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(city_data["train"][numeric_cols])
    X_val = scaler.transform(city_data["val"][numeric_cols])
    X_test = scaler.transform(city_data["test"][numeric_cols])

    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i : i + sequence_length])
        return np.array(sequences)

    X_train_seq = create_sequences(X_train, sequence_length)
    X_val_seq = create_sequences(X_val, sequence_length)
    X_test_seq = create_sequences(X_test, sequence_length)
    y_train = city_data["train"]["AQI"].values[sequence_length - 1 :]
    y_val = city_data["val"]["AQI"].values[sequence_length - 1 :]
    y_test = city_data["test"]["AQI"].values[sequence_length - 1 :]
    # Target scaling
    if target_scaler is None:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    X_train = torch.FloatTensor(X_train_seq)
    y_train = torch.FloatTensor(y_train_scaled)
    X_val = torch.FloatTensor(X_val_seq)
    y_val = torch.FloatTensor(y_val_scaled)
    X_test = torch.FloatTensor(X_test_seq)
    y_test = torch.FloatTensor(y_test_scaled)
    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        len(numeric_cols),
        target_scaler,
        y_train,
        y_val,
        y_test,
    )


def calculate_lstm_feature_importance(
    model, X_val, y_val, feature_names, n_permutations=10
):
    """Calculate permutation importance for LSTM model"""
    device = next(model.parameters()).device
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Get baseline performance
    with torch.no_grad():
        baseline_pred = model(X_val)
    baseline_rmse = torch.sqrt(torch.mean((baseline_pred.squeeze() - y_val) ** 2))

    # Calculate importance for each feature
    importance = {}
    for i, feature in enumerate(feature_names):
        # Store original performance changes
        performance_changes = []

        for _ in range(n_permutations):
            # Create perturbed input by shuffling the feature
            X_perturbed = X_val.clone()
            shuffled_indices = torch.randperm(X_perturbed.size(0))
            X_perturbed[:, :, i] = X_perturbed[shuffled_indices, :, i]

        # Get prediction with perturbed input
        with torch.no_grad():
            perturbed_pred = model(X_perturbed)

            # Calculate performance change
            perturbed_rmse = torch.sqrt(
                torch.mean((perturbed_pred.squeeze() - y_val) ** 2)
            )
            performance_changes.append((perturbed_rmse - baseline_rmse).item())

        # Importance is the mean performance degradation
        importance[feature] = np.mean(performance_changes)

    return importance


def plot_feature_importance(feature_importance, city_name):
    """Plot feature importance scores"""
    # Sort features by importance
    sorted_features = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Create the plot
    plt.figure(figsize=(12, 6))
    features = list(sorted_features.keys())
    importance = list(sorted_features.values())

    # Create horizontal bar plot
    plt.barh(range(len(features)), importance, align="center")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Feature Importance Score")
    plt.title(f"Feature Importance for {city_name.title()}")
    plt.tight_layout()

    return plt.gcf()


def save_lstm_results(city_name, results, feature_importance, model):
    """Save LSTM model results and visualizations"""
    # Create results directory
    results_dir = f"./results/lstm/{city_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics = {"validation": results["val_metrics"], "test": results["test_metrics"]}

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save feature importance
    with open(f"{results_dir}/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=4)

    # Save model
    torch.save(model.state_dict(), f"{results_dir}/model.pth")

    # Generate and save visualizations
    plot_metrics_comparison(city_name, results)
    plt.savefig(f"{results_dir}/metrics_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Plot and save feature importance
    plot_feature_importance(feature_importance, city_name)
    plt.savefig(f"{results_dir}/feature_importance.png", bbox_inches="tight", dpi=300)
    plt.close()

    for split in ["val", "test"]:
        plot_predictions(city_name, results, split, results["actual_values"][split])
        plt.savefig(
            f"{results_dir}/predictions_{split}.png", bbox_inches="tight", dpi=300
        )
        plt.close()


def plot_metrics_comparison(city, results):
    """Plot comparison of metrics for different models"""
    # Define metrics and splits
    metrics = ["rmse", "mae", "r2"]
    splits = ["val", "test"]
    split_to_metrics_key = {"val": "val_metrics", "test": "test_metrics"}

    # Create figure
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = []
        labels = []
        for split in splits:
            metrics_key = split_to_metrics_key[split]
            if metrics_key in results and metric in results[metrics_key]:
                values.append(results[metrics_key][metric])
                labels.append(split.capitalize())
        x = np.arange(1)  # Only one model (LSTM)
        width = 0.35
        for j, (val, label) in enumerate(zip(values, labels)):
            ax.bar(x + j * width, [val], width, label=label)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(["LSTM"])
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Model Performance Comparison - {city.title()}", y=1.05)
    plt.tight_layout()
    return fig


def plot_predictions(city, results, data_split, actual_values):
    """Plot actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot actual values
    ax.plot(actual_values, label="Actual", color="black", alpha=0.7)

    # Plot predictions
    predictions = results["predictions"][data_split]
    ax.plot(predictions, label="LSTM", alpha=0.7)

    # Customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("AQI")
    ax.set_title(f"Actual vs Predicted AQI - {data_split.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def plot_pred_vs_actual_scatter(predictions, actuals, city, split):
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], "r--")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Predicted vs Actual AQI Values ({city.title()} - {split})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def print_feature_variance(city_data, split="train"):
    print(f"\nFeature variance for {split} set:")
    numeric_cols = city_data[split].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        var = city_data[split][col].var()
        print(f"  {col}: variance={var:.4f}")


def check_target_scaling(city_data, split="train"):
    print(f"\nTarget (AQI) stats for {split} set:")
    y = city_data[split]["AQI"].values
    print(
        f"  min={np.min(y):.2f}, max={np.max(y):.2f}, mean={np.mean(y):.2f}, std={np.std(y):.2f}"
    )


def check_sequence_alignment(city_data, sequence_length=7, split="train"):
    print(f"\nChecking sequence/target alignment for {split} set:")
    numeric_cols = city_data[split].select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "AQI"]
    X = city_data[split][numeric_cols].values
    y = city_data[split]["AQI"].values
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    if len(y) > sequence_length:
        print(f"  First sequence features (first row): {X[0]}")
        print(
            f"  Corresponding target (at index {sequence_length-1}): {y[sequence_length-1]}"
        )
        print(f"  Next target: {y[sequence_length]}")


def overfit_small_batch(city_data, sequence_length=7, batch_size=8, epochs=200):
    print("\n--- Overfit Small Batch Test ---")
    (
        (X_train, y_train),
        _,
        _,
        input_size,
        target_scaler,
        y_train_orig,
        _,
        _,
    ) = prepare_lstm_data(city_data, sequence_length=sequence_length)
    X_small = X_train[:batch_size]
    y_small = y_train[:batch_size]
    y_small_orig = y_train_orig[:batch_size]
    model = LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=1).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_small).squeeze(-1)
        if epoch == 0:
            print(
                f"[LSTM] outputs.shape: {outputs.shape}, y_small.shape: {y_small.shape}"
            )
        loss = criterion(outputs, y_small)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    print(f"Final loss after {epochs} epochs: {losses[-1]:.4f}")
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Overfit Small Batch Loss Curve (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    # Print predictions vs actuals (inverse transform)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_small).squeeze(-1).numpy().flatten()
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    print("Predictions vs Actuals (small batch, LSTM):")
    for p, a in zip(preds, y_small_orig):
        print(f"  Pred: {p:.2f}, Actual: {a:.2f}")
    # Linear regression sanity check
    print("\n--- Linear Regression Overfit Test ---")
    X_small_np = X_small[:, -1, :].numpy()  # Use last time step as features
    linreg = LinearRegression()
    linreg.fit(X_small_np, y_small_orig)
    preds_lr = linreg.predict(X_small_np)
    print("Predictions vs Actuals (small batch, Linear Regression):")
    for p, a in zip(preds_lr, y_small_orig):
        print(f"  Pred: {p:.2f}, Actual: {a:.2f}")


# Cell 4: Main execution
def main():
    cities = ["delhi", "hyderabad", "chennai", "bengaluru"]
    for city in cities:
        print(f"\nProcessing {city}...")
        city_data = load_city_data(city)
        if not city_data:
            print(f"No data found for {city}")
            continue
        # --- Diagnostics before training ---
        print_feature_variance(city_data, split="train")
        check_target_scaling(city_data, split="train")
        check_sequence_alignment(city_data, sequence_length=7, split="train")
        overfit_small_batch(city_data, sequence_length=7, batch_size=8, epochs=200)
        # Train and evaluate LSTM model
        results = train_and_evaluate_lstm(
            city_data=city_data,
            city_name=city,
            epochs=200,
            batch_size=32,
            lr=0.001,
            hidden_size=128,
            num_layers=2,
        )
        print("\nModel Performance Summary:")
        print("Validation Set:")
        print(f"  RMSE: {results['val_metrics']['rmse']:.2f}")
        print(f"  MAE: {results['val_metrics']['mae']:.2f}")
        print(f"  R²: {results['val_metrics']['r2']:.2f}")
        print("\nTest Set:")
        print(f"  RMSE: {results['test_metrics']['rmse']:.2f}")
        print(f"  MAE: {results['test_metrics']['mae']:.2f}")
        print(f"  R²: {results['test_metrics']['r2']:.2f}")


def train_and_evaluate_lstm(
    city_data,
    city_name,
    epochs=200,
    batch_size=32,
    lr=0.001,
    hidden_size=128,
    num_layers=2,
):
    print(f"\n🔄 Preprocessing data for {city_name}...")
    (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        input_size,
        target_scaler,
        y_train_orig,
        y_val_orig,
        y_test_orig,
    ) = prepare_lstm_data(city_data)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    print("🔧 Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print("🏃 Starting training...")
    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze(-1)
            if epoch == 0:
                print(
                    f"[LSTM] outputs.shape: {outputs.shape}, batch_y.shape: {batch_y.shape}"
                )
            optimizer.zero_grad()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze(-1)
                val_loss += criterion(outputs, batch_y).item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_true = []
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze(-1)
            val_preds.extend(outputs.cpu().numpy())
            val_true.extend(batch_y.cpu().numpy())
        test_preds = []
        test_true = []
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze(-1)
            test_preds.extend(outputs.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())
    # Inverse transform predictions and targets
    val_preds_inv = target_scaler.inverse_transform(
        np.array(val_preds).reshape(-1, 1)
    ).flatten()
    val_true_inv = target_scaler.inverse_transform(
        np.array(val_true).reshape(-1, 1)
    ).flatten()
    test_preds_inv = target_scaler.inverse_transform(
        np.array(test_preds).reshape(-1, 1)
    ).flatten()
    test_true_inv = target_scaler.inverse_transform(
        np.array(test_true).reshape(-1, 1)
    ).flatten()
    val_metrics = {
        "rmse": np.sqrt(mean_squared_error(val_true_inv, val_preds_inv)),
        "mae": mean_absolute_error(val_true_inv, val_preds_inv),
        "r2": r2_score(val_true_inv, val_preds_inv),
    }
    test_metrics = {
        "rmse": np.sqrt(mean_squared_error(test_true_inv, test_preds_inv)),
        "mae": mean_absolute_error(test_true_inv, test_preds_inv),
        "r2": r2_score(test_true_inv, test_preds_inv),
    }
    numeric_cols = city_data["train"].select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "AQI"]
    feature_importance = calculate_lstm_feature_importance(
        model, X_val, y_val, numeric_cols
    )
    results = {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "predictions": {"val": val_preds_inv, "test": test_preds_inv},
        "actual_values": {"val": val_true_inv, "test": test_true_inv},
    }
    save_lstm_results(city_name, results, feature_importance, model)
    print(f"\nDiagnostics for {city_name} (Test Set):")
    test_preds_arr = np.array(test_preds_inv)
    test_true_arr = np.array(test_true_inv)
    print(
        f"Predictions: min={test_preds_arr.min():.2f}, max={test_preds_arr.max():.2f}, mean={test_preds_arr.mean():.2f}, std={test_preds_arr.std():.2f}"
    )
    print(
        f"Actuals: min={test_true_arr.min():.2f}, max={test_true_arr.max():.2f}, mean={test_true_arr.mean():.2f}, std={test_true_arr.std():.2f}"
    )
    # plot_pred_vs_actual_scatter(test_preds_arr, test_true_arr, city_name, 'test')
    # plot_loss_curves(train_losses, val_losses)
    return results


if __name__ == "__main__":
    main()
