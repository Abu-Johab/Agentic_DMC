import os
import random
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CNN1D(nn.Module):
    def __init__(self, in_bands=204, f1=32, f2=128, k1=5, k2=3, d1=0.1, d2=0.2, dd=0.1, du=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, f1, k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(f1, f2, k2, padding=k2 // 2)
        self.bn1   = nn.BatchNorm1d(f1)
        self.bn2   = nn.BatchNorm1d(f2)
        self.pool  = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(d1)
        self.drop2 = nn.Dropout(d2)
        self.dropd = nn.Dropout(dd)
        self.act   = nn.ReLU()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_bands)
            out_dim = self._features(dummy).shape[1]
        self.fc1 = nn.Linear(out_dim, du)
        self.fc2 = nn.Linear(du, 1)

    def _features(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x):
        x = self._features(x)
        x = self.dropd(self.act(self.fc1(x)))
        return self.fc2(x)

class CNNWrapper:
    """Top-level serializable wrapper for PyTorch CNN1D model."""
    def __init__(self, torch_model, y_mean, y_std, device):
        self.torch_model = torch_model
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device
        
    def predict(self, X):
        self.torch_model.eval()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            p_s = self.torch_model(X_t).cpu().numpy().ravel()
        return p_s * self.y_std + self.y_mean

class ModelingTool:
    """
    Scientific ML Model Engine:
    Handles PLSR, SVR (StandardScaler + RBF, without PCA), Random Forest, and 1D CNN.
    """
    
    @staticmethod
    def train_plsr(Xc, yc, Xv, yv, max_lv=10, seed=42):
        set_seed(seed)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=seed)
        rmsecv_list = []
        
        for lv in range(1, max_lv + 1):
            pred, cnt = np.zeros_like(yc), np.zeros_like(yc)
            for tr, va in cv.split(Xc):
                m = PLSRegression(n_components=lv, scale=False).fit(Xc[tr], yc[tr])
                pred[va] += m.predict(Xc[va]).ravel()
                cnt[va] += 1
            rmse_val = float(np.sqrt(mean_squared_error(yc, pred / cnt)))
            rmsecv_list.append(rmse_val)
            
        best_lv = int(np.argmin(rmsecv_list)) + 1
        best_rmsecv = float(min(rmsecv_list))
        
        model = PLSRegression(n_components=best_lv, scale=False).fit(Xc, yc)
        pred_val = model.predict(Xv).ravel()
        rmsev = float(np.sqrt(mean_squared_error(yv, pred_val)))
        
        return {
            "model_type": "PLSR",
            "model": model,
            "best_params": {"n_components": best_lv},
            "rmsecv": best_rmsecv,
            "rmsev": rmsev,
            "rmsecv_curve": rmsecv_list
        }

    @staticmethod
    def train_svr(Xc, yc, Xv, yv, seed=42):
        set_seed(seed)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf"))
        ])
        grid = {
            "svr__C": [0.5, 1.0, 2.0, 5.0, 10.0],
            "svr__gamma": ["scale", "auto", 1e-3, 5e-3],
            "svr__epsilon": [0.03, 0.05, 0.08]
        }
        gs = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=seed),
                          scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xc, yc)
                          
        best_rmsecv = float(-gs.best_score_)
        best_model = gs.best_estimator_
        pred_val = best_model.predict(Xv)
        rmsev = float(np.sqrt(mean_squared_error(yv, pred_val)))
        
        return {
            "model_type": "SVR",
            "model": best_model,
            "best_params": gs.best_params_,
            "rmsecv": best_rmsecv,
            "rmsev": rmsev
        }

    @staticmethod
    def train_rf(Xc, yc, Xv, yv, n_estimators=100, seed=42):
        set_seed(seed)
        grid = {
            "max_depth": [3, 5, 8],
            "min_samples_leaf": [3, 5, 10],
            "max_features": [0.05, 0.1, 0.2],
            "max_samples": [0.5, 0.7, 1.0]
        }
        gs = GridSearchCV(RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1),
                          grid, cv=KFold(5, shuffle=True, random_state=seed),
                          scoring="neg_root_mean_squared_error", n_jobs=1).fit(Xc, yc)
                          
        best_rmsecv = float(-gs.best_score_)
        best_params = gs.best_params_
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            max_samples=best_params["max_samples"],
            n_jobs=-1
        ).fit(Xc, yc)
        
        pred_val = rf.predict(Xv)
        rmsev = float(np.sqrt(mean_squared_error(yv, pred_val)))
        
        return {
            "model_type": "RandomForest",
            "model": rf,
            "best_params": best_params,
            "rmsecv": best_rmsecv,
            "rmsev": rmsev
        }

    @staticmethod
    def train_cnn1d(Xc, yc, Xv, yv, epochs=100, patience=15, seed=42):
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        y_mean, y_std = float(np.mean(yc)), float(np.std(yc))
        if y_std == 0: y_std = 1.0
        yc_s = (yc - y_mean) / y_std
        yv_s = (yv - y_mean) / y_std
        
        ds_tr = TensorDataset(torch.from_numpy(Xc).float().unsqueeze(1), torch.from_numpy(yc_s).float().unsqueeze(1))
        ds_va = TensorDataset(torch.from_numpy(Xv).float().unsqueeze(1), torch.from_numpy(yv_s).float().unsqueeze(1))
        
        loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
        loader_va = DataLoader(ds_va, batch_size=16, shuffle=False)
        
        in_bands = Xc.shape[1]
        model = CNN1D(in_bands=in_bands).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        best_val_loss = float("inf")
        best_state = None
        patience_count = 0
        history = []
        
        for ep in range(epochs):
            model.train()
            tr_loss = 0.0
            for xb, yb in loader_tr:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(Xc)
            
            model.eval()
            val_preds = []
            with torch.no_grad():
                for xb, _ in loader_va:
                    val_preds.append(model(xb.to(device)).cpu())
            pv_s = torch.cat(val_preds).numpy().ravel()
            pv = pv_s * y_std + y_mean
            val_rmse = float(np.sqrt(mean_squared_error(yv, pv)))
            
            history.append({"epoch": ep + 1, "train_loss": tr_loss, "val_rmse": val_rmse})
            
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break
                    
        model.load_state_dict(best_state)
        wrapper = CNNWrapper(model, y_mean, y_std, device)
        
        return {
            "model_type": "CNN1D",
            "model": wrapper,
            "best_params": {"epochs_trained": len(history), "best_val_rmse": best_val_loss},
            "rmsecv": best_val_loss,
            "rmsev": best_val_loss,
            "training_history": history
        }
