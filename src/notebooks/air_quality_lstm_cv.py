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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_palette("viridis")

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


# Reuse the LSTM model class from the original file
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


def load_cv_data(city_name, fold_num):
    """Load cross-validation data for a specific city and fold"""
    data_dir = "../data/processed/cv_splits"
    fold_dir = f"{data_dir}/{city_name}/fold_{fold_num}"

    train_data = pd.read_csv(f"{fold_dir}/train.csv")
    test_data = pd.read_csv(f"{fold_dir}/test.csv")

    train_data["Date"] = pd.to_datetime(train_data["Date"])
    test_data["Date"] = pd.to_datetime(test_data["Date"])

    return {"train": train_data, "test": test_data}


def prepare_lstm_data(city_data, sequence_length=7, target_scaler=None):
    """Prepare data for LSTM model with sequence creation and target scaling"""
    numeric_cols = city_data["train"].select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "AQI"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(city_data["train"][numeric_cols])
    X_test = scaler.transform(city_data["test"][numeric_cols])

    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i : i + sequence_length])
        return np.array(sequences)

    X_train_seq = create_sequences(X_train, sequence_length)
    X_test_seq = create_sequences(X_test, sequence_length)

    y_train = city_data["train"]["AQI"].values[sequence_length - 1 :]
    y_test = city_data["test"]["AQI"].values[sequence_length - 1 :]

    # Target scaling
    if target_scaler is None:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()

    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_train = torch.FloatTensor(X_train_seq)
    y_train = torch.FloatTensor(y_train_scaled)
    X_test = torch.FloatTensor(X_test_seq)
    y_test = torch.FloatTensor(y_test_scaled)

    return (
        (X_train, y_train),
        (X_test, y_test),
        len(numeric_cols),
        target_scaler,
        y_train,
        y_test,
    )


def train_and_evaluate_lstm_cv(
    city_data,
    city_name,
    fold_num,
    epochs=200,
    batch_size=32,
    lr=0.001,
    hidden_size=128,
    num_layers=2,
):
    print(f"\n🔄 Preprocessing data for {city_name} (Fold {fold_num})...")
    (
        (X_train, y_train),
        (X_test, y_test),
        input_size,
        target_scaler,
        y_train_orig,
        y_test_orig,
    ) = prepare_lstm_data(city_data)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
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
    best_loss = float("inf")
    patience = 15
    patience_counter = 0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze(-1)
            optimizer.zero_grad()
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step(avg_train_loss)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_true = []
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze(-1)
            test_preds.extend(outputs.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())

    # Inverse transform predictions and targets
    test_preds_inv = target_scaler.inverse_transform(
        np.array(test_preds).reshape(-1, 1)
    ).flatten()
    test_true_inv = target_scaler.inverse_transform(
        np.array(test_true).reshape(-1, 1)
    ).flatten()

    metrics = {
        "rmse": np.sqrt(mean_squared_error(test_true_inv, test_preds_inv)),
        "mae": mean_absolute_error(test_true_inv, test_preds_inv),
        "r2": r2_score(test_true_inv, test_preds_inv),
    }

    results = {
        "fold": fold_num,
        "metrics": metrics,
        "predictions": test_preds_inv,
        "actual_values": test_true_inv,
        "train_losses": train_losses,
    }

    # Save results
    save_cv_results(city_name, fold_num, results, model)

    return results


def save_cv_results(city_name, fold_num, results, model):
    """Save cross-validation results and model"""
    results_dir = f"./results/lstm_cv/{city_name}/fold_{fold_num}"
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=4)

    # Save model
    torch.save(model.state_dict(), f"{results_dir}/model.pth")

    # Generate and save visualizations
    plot_cv_predictions(city_name, fold_num, results)
    plt.savefig(f"{results_dir}/predictions.png", bbox_inches="tight", dpi=300)
    plt.close()

    plot_cv_loss_curve(results["train_losses"])
    plt.savefig(f"{results_dir}/loss_curve.png", bbox_inches="tight", dpi=300)
    plt.close()


def plot_cv_predictions(city, fold_num, results):
    """Plot actual vs predicted values for CV fold"""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot actual values
    ax.plot(results["actual_values"], label="Actual", color="black", alpha=0.7)

    # Plot predictions
    ax.plot(results["predictions"], label="LSTM", alpha=0.7)

    # Customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("AQI")
    ax.set_title(f"Actual vs Predicted AQI - {city.title()} (Fold {fold_num})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cv_loss_curve(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def main():
    cities = ["delhi", "hyderabad", "chennai", "bengaluru"]
    n_folds = 3

    for city in cities:
        print(f"\nProcessing {city}...")
        fold_results = []

        for fold in range(1, n_folds + 1):
            print(f"\nTraining fold {fold}...")
            city_data = load_cv_data(city, fold)

            results = train_and_evaluate_lstm_cv(
                city_data=city_data,
                city_name=city,
                fold_num=fold,
                epochs=200,
                batch_size=32,
                lr=0.001,
                hidden_size=128,
                num_layers=2,
            )

            fold_results.append(results)

            print(f"\nFold {fold} Results:")
            print(f"RMSE: {results['metrics']['rmse']:.2f}")
            print(f"MAE: {results['metrics']['mae']:.2f}")
            print(f"R²: {results['metrics']['r2']:.2f}")

        # Calculate average metrics across folds
        avg_metrics = {
            "rmse": np.mean([r["metrics"]["rmse"] for r in fold_results]),
            "mae": np.mean([r["metrics"]["mae"] for r in fold_results]),
            "r2": np.mean([r["metrics"]["r2"] for r in fold_results]),
        }

        # Calculate standard deviation of metrics across folds
        std_metrics = {
            "rmse": np.std([r["metrics"]["rmse"] for r in fold_results]),
            "mae": np.std([r["metrics"]["mae"] for r in fold_results]),
            "r2": np.std([r["metrics"]["r2"] for r in fold_results]),
        }

        print(f"\nAverage Metrics for {city}:")
        print(f"RMSE: {avg_metrics['rmse']:.2f} ± {std_metrics['rmse']:.2f}")
        print(f"MAE: {avg_metrics['mae']:.2f} ± {std_metrics['mae']:.2f}")
        print(f"R²: {avg_metrics['r2']:.2f} ± {std_metrics['r2']:.2f}")

        # Save aggregated metrics to a JSON file
        aggregated_results = {
            "city": city,
            "avg_metrics": avg_metrics,
            "std_metrics": std_metrics,
        }
        results_dir = f"./results/lstm_cv/{city}"
        os.makedirs(results_dir, exist_ok=True)
        with open(f"{results_dir}/aggregated_metrics.json", "w") as f:
            json.dump(aggregated_results, f, indent=4)


if __name__ == "__main__":
    main()
