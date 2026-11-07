import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

class PreprocessingTool:
    """
    Deterministic preprocessing transformers:
    Supports leakage-free fitting of MSC reference and scalers.
    
    Allowed candidate methods:
    - raw
    - snv
    - msc
    - sg_smooth
    - sg_deriv1
    - snv_sg1
    - msc_sg1
    """
    
    def __init__(self, method="snv_sg1", sg_window=11, sg_poly=2):
        self.method = method
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.msc_ref = None
        self.scaler = None
        self.fitted = False

    @staticmethod
    def snv(X):
        """Standard Normal Variate (per spectrum)."""
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-8
        return (X - mean) / std

    def fit(self, X_train):
        """Fit parameters (e.g. MSC mean reference spectrum) on calibration set."""
        X_tr = np.asarray(X_train, dtype=np.float64)
        if "msc" in self.method:
            # MSC reference spectrum is the mean spectrum of training data
            self.msc_ref = np.mean(X_tr, axis=0)
        self.fitted = True
        return self

    def apply_msc(self, X):
        if self.msc_ref is None:
            raise ValueError("MSC transformer has not been fitted with training data!")
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            fit = np.polyfit(self.msc_ref, X[i], 1) # fit slope and intercept
            X_msc[i] = (X[i] - fit[1]) / (fit[0] + 1e-8)
        return X_msc

    def transform(self, X):
        X_proc = np.asarray(X, dtype=np.float64).copy()
        
        # 1D array handling
        is_1d = (X_proc.ndim == 1)
        if is_1d:
            X_proc = X_proc.reshape(1, -1)

        if self.method == "raw":
            pass
        elif self.method == "snv":
            X_proc = self.snv(X_proc)
        elif self.method == "msc":
            X_proc = self.apply_msc(X_proc)
        elif self.method == "sg_smooth":
            X_proc = savgol_filter(X_proc, window_length=self.sg_window, polyorder=self.sg_poly, deriv=0, axis=1)
        elif self.method == "sg_deriv1":
            X_proc = savgol_filter(X_proc, window_length=self.sg_window, polyorder=self.sg_poly, deriv=1, axis=1)
        elif self.method == "snv_sg1":
            X_snv = self.snv(X_proc)
            X_proc = savgol_filter(X_snv, window_length=self.sg_window, polyorder=self.sg_poly, deriv=1, axis=1)
        elif self.method == "msc_sg1":
            X_msc = self.apply_msc(X_proc)
            X_proc = savgol_filter(X_msc, window_length=self.sg_window, polyorder=self.sg_poly, deriv=1, axis=1)
        else:
            raise ValueError(f"Unsupported preprocessing method: {self.method}")

        if is_1d:
            return X_proc.ravel()
        return X_proc

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
