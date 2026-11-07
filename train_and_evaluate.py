"""
Model Training & Full Performance Evaluation Suite (Agentic DMC)
Trains candidate models (PLSR, SVR, Random Forest, 1D-CNN) from scratch on Calibration,
evaluates on Validation and Prediction (Test) sets, and compares against the Deployed Frozen Model.

Computes and prints:
- Calibration: R2, RMSE (g/g), RPD, MAE (g/g), RPIQ
- Validation:  R2, RMSE (g/g), RPD, MAE (g/g), RPIQ
- Prediction:  R2, RMSE (g/g), RPD, MAE (g/g), RPIQ
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

WORKSPACE_ROOT = os.path.normpath(os.path.dirname(__file__))

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    sd_true = float(np.std(y_true, ddof=1))
    rpd = float(sd_true / rmse) if rmse > 0 else 0.0
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = float((q75 - q25) / rmse) if rmse > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "rpd": rpd, "rpiq": rpiq}

def apply_snv_sg1(X, window=11, poly=2):
    """Standard Normal Variate (SNV) followed by Savitzky-Golay 1st derivative."""
    X_in = np.array(X, dtype=float).copy()
    mean = np.mean(X_in, axis=1, keepdims=True)
    std = np.std(X_in, axis=1, keepdims=True)
    std[std == 0] = 1.0
    X_snv = (X_in - mean) / std
    return savgol_filter(X_snv, window_length=window, polyorder=poly, deriv=1, axis=1)

# 1D-CNN Architecture
class CNN1D(nn.Module):
    def __init__(self, in_features=204):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        pooled_dim = 128 * (in_features // 4)
        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

def train_cnn1d_model(X_train, y_train, X_val, y_val, epochs=60, lr=0.001, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train, ddof=1))
    if y_std == 0: y_std = 1.0
    
    y_tr_norm = (y_train - y_mean) / y_std
    y_va_norm = (y_val - y_mean) / y_std
    
    t_X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    t_y_tr = torch.tensor(y_tr_norm, dtype=torch.float32).unsqueeze(1)
    t_X_va = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    t_y_va = torch.tensor(y_va_norm, dtype=torch.float32).unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(t_X_tr, t_y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = CNN1D(in_features=X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_weights = None
    patience, p_count = 15, 0
    
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(t_X_va), t_y_va).item()
            
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            p_count = 0
        else:
            p_count += 1
            if p_count >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    
    def predict_fn(X_eval):
        t_X_eval = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            preds_norm = model(t_X_eval).squeeze(1).numpy()
        return preds_norm * y_std + y_mean
        
    return predict_fn

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models for Dry Matter Content prediction")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs for 1D-CNN training")
    parser.add_argument("--skip-cnn-train", action="store_true", help="Skip fresh CNN training to evaluate faster")
    args = parser.parse_args()

    print("=" * 90)
    print("      AGENTIC DMC: MODEL TRAINING & METRICS EVALUATION PIPELINE")
    print("=" * 90)

    # 1. Resolve Dataset and Partition Split Paths
    # FIX R-06: Prioritize local directory split_csv
    dataset_csv = os.path.join(WORKSPACE_ROOT, "SpectroFood_dataset.csv")
    split_csv = os.path.join(WORKSPACE_ROOT, "plsr_split_ks70_15_15.csv")
    if not os.path.exists(split_csv):
        split_csv = r"E:\Thesis 2025\Agent_mushroom\plsr_split_ks70_15_15.csv"
    if not os.path.exists(split_csv):
        split_csv = os.path.join(WORKSPACE_ROOT, "..", "Agent_mushroom", "plsr_split_ks70_15_15.csv")

    if not os.path.exists(dataset_csv):
        print(f"[ERROR] Dataset file not found at: {dataset_csv}")
        sys.exit(1)
    if not os.path.exists(split_csv):
        print(f"[ERROR] Split file not found at: {split_csv}")
        sys.exit(1)

    print(f"Loading Dataset: {dataset_csv}")
    print(f"Loading Splits : {split_csv}")

    df = pd.read_csv(dataset_csv)
    sp_df = pd.read_csv(split_csv).set_index("Mush No").loc[df["Mush No"]].reset_index()

    wave_cols = [c for c in df.columns if c not in ["Mush No", "Dry matter"]]
    X_raw = df[wave_cols].values.astype(np.float64)
    y = df["Dry matter"].values.astype(np.float64)

    cal_mask = (sp_df["split"] == "cal").values
    val_mask = (sp_df["split"] == "val").values
    tst_mask = (sp_df["split"] == "test").values

    X_cal_raw, y_cal = X_raw[cal_mask], y[cal_mask]
    X_val_raw, y_val = X_raw[val_mask], y[val_mask]
    X_tst_raw, y_tst = X_raw[tst_mask], y[tst_mask]

    print(f"Data Partitions: Calibration N={len(X_cal_raw)}, Validation N={len(X_val_raw)}, Prediction/Test N={len(X_tst_raw)}")
    print(f"Applying Preprocessing: SNV + Savitzky-Golay 1st Derivative (snv_sg1)...")
    
    X_cal = apply_snv_sg1(X_cal_raw)
    X_val = apply_snv_sg1(X_val_raw)
    X_tst = apply_snv_sg1(X_tst_raw)

    results_table = []

    def evaluate_model(name, predict_func):
        pred_cal = predict_func(X_cal)
        pred_val = predict_func(X_val)
        pred_tst = predict_func(X_tst)

        m_cal = compute_metrics(y_cal, pred_cal)
        m_val = compute_metrics(y_val, pred_val)
        m_tst = compute_metrics(y_tst, pred_tst)

        results_table.append({"model": name, "split": "Calibration", "n": len(y_cal), **m_cal})
        results_table.append({"model": name, "split": "Validation",  "n": len(y_val), **m_val})
        results_table.append({"model": name, "split": "Prediction/Test", "n": len(y_tst), **m_tst})

        print(f"\n[{name}]")
        print(f"  Calibration     (N={len(y_cal)}): R2 = {m_cal['r2']:.4f} | RMSE = {m_cal['rmse']:.4f} g/g | RPD = {m_cal['rpd']:.4f} | MAE = {m_cal['mae']:.4f} g/g")
        print(f"  Validation      (N={len(y_val)}): R2 = {m_val['r2']:.4f} | RMSE = {m_val['rmse']:.4f} g/g | RPD = {m_val['rpd']:.4f} | MAE = {m_val['mae']:.4f} g/g")
        print(f"  Prediction/Test (N={len(y_tst)}): R2 = {m_tst['r2']:.4f} | RMSE = {m_tst['rmse']:.4f} g/g | RPD = {m_tst['rpd']:.4f} | MAE = {m_tst['mae']:.4f} g/g")

    # --- 1. Train PLSR ---
    print("\n[1/5] Training Partial Least Squares Regression (PLSR, 10 Latent Variables)...")
    plsr = PLSRegression(n_components=10)
    plsr.fit(X_cal, y_cal)
    evaluate_model("PLSR (10 LVs)", lambda x: plsr.predict(x).ravel())

    # --- 2. Train SVR ---
    # FIX R-04: SVR trained directly on standardized spectra without PCA (matching modeling.py and discuss.md Section 1)
    print("\n[2/5] Training Support Vector Regression (StandardScaler + RBF Kernel, without PCA)...")
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(C=5.0, kernel='rbf', epsilon=0.01))
    ])
    svr_pipe.fit(X_cal, y_cal)
    evaluate_model("SVR (StandardScaler+RBF)", lambda x: svr_pipe.predict(x).ravel())

    # --- 3. Train Random Forest ---
    print("\n[3/5] Training Random Forest Regressor (100 Trees)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=4)
    rf.fit(X_cal, y_cal)
    evaluate_model("Random Forest", lambda x: rf.predict(x).ravel())

    # --- 4. Train 1D-CNN (PyTorch) ---
    if not args.skip_cnn_train:
        print(f"\n[4/5] Training PyTorch 1D-CNN Model from scratch ({args.epochs} max epochs with early stopping)...")
        cnn_predict_fn = train_cnn1d_model(X_cal, y_cal, X_val, y_val, epochs=args.epochs, seed=42)
        evaluate_model("1D-CNN (Freshly Trained)", cnn_predict_fn)
    else:
        print("\n[4/5] Skipping fresh 1D-CNN training (--skip-cnn-train set).")

    # --- 5. Evaluate Deployed Frozen Model Checkpoint ---
    ckpt_path = os.path.join(WORKSPACE_ROOT, "agentic_dmc", "registry", "frozen_pipeline.pkl")
    if os.path.exists(ckpt_path):
        print(f"\n[5/5] Evaluating Deployed Frozen Pipeline Checkpoint: {ckpt_path}...")
        with open(ckpt_path, "rb") as f:
            pkg = pickle.load(f)
        frozen_model = pkg["model"]
        frozen_preproc = pkg["preprocessor"]
        
        def frozen_predict_fn(x_raw):
            x_p = frozen_preproc.transform(x_raw)
            return frozen_model.predict(x_p).ravel()

        pred_cal_f = frozen_predict_fn(X_cal_raw)
        pred_val_f = frozen_predict_fn(X_val_raw)
        pred_tst_f = frozen_predict_fn(X_tst_raw)

        m_cal_f = compute_metrics(y_cal, pred_cal_f)
        m_val_f = compute_metrics(y_val, pred_val_f)
        m_tst_f = compute_metrics(y_tst, pred_tst_f)

        results_table.append({"model": "1D-CNN (Deployed Frozen)", "split": "Calibration", "n": len(y_cal), **m_cal_f})
        results_table.append({"model": "1D-CNN (Deployed Frozen)", "split": "Validation",  "n": len(y_val), **m_val_f})
        results_table.append({"model": "1D-CNN (Deployed Frozen)", "split": "Prediction/Test", "n": len(y_tst), **m_tst_f})

        print(f"\n[1D-CNN (Deployed Frozen Package fe8aa98e...)]")
        print(f"  Calibration     (N={len(y_cal)}): R2 = {m_cal_f['r2']:.4f} | RMSE = {m_cal_f['rmse']:.4f} g/g | RPD = {m_cal_f['rpd']:.4f} | MAE = {m_cal_f['mae']:.4f} g/g")
        print(f"  Validation      (N={len(y_val)}): R2 = {m_val_f['r2']:.4f} | RMSE = {m_val_f['rmse']:.4f} g/g | RPD = {m_val_f['rpd']:.4f} | MAE = {m_val_f['mae']:.4f} g/g")
        print(f"  Prediction/Test (N={len(y_tst)}): R2 = {m_tst_f['r2']:.4f} | RMSE = {m_tst_f['rmse']:.4f} g/g | RPD = {m_tst_f['rpd']:.4f} | MAE = {m_tst_f['mae']:.4f} g/g")

    # --- Comprehensive Summary Table ---
    print("\n" + "=" * 105)
    print("                                MASTER PERFORMANCE EVALUATION MATRIX")
    print("=" * 105)
    header = f"{'Model Architecture':<26} | {'Split':<15} | {'N':<4} | {'R2':<7} | {'RMSE (g/g)':<10} | {'RPD':<7} | {'MAE (g/g)':<10} | {'RPIQ':<7}"
    print(header)
    print("-" * 105)

    last_model = None
    for r in results_table:
        m_name = r["model"] if r["model"] != last_model else ""
        last_model = r["model"]
        row = f"{m_name:<26} | {r['split']:<15} | {r['n']:<4} | {r['r2']:<7.4f} | {r['rmse']:<10.4f} | {r['rpd']:<7.4f} | {r['mae']:<10.4f} | {r['rpiq']:<7.4f}"
        print(row)
        if r["split"] == "Prediction/Test":
            print("-" * 105)

    print("=" * 105)
    print("Done! Evaluation completed successfully.")

if __name__ == "__main__":
    main()
