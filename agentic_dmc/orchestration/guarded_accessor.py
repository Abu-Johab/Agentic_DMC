"""
Guarded Holdout Accessor
Enforces Global Rule 3: The 37-sample holdout may be used EXACTLY ONCE,
at the very end, after all model selection and hyperparameter tuning is finished.
Raises RuntimeError on early or repeated access attempts.
"""

import numpy as np
import pandas as pd

class GuardedHoldoutAccessor:
    _accessed = False
    _access_log = []

    @classmethod
    def reset(cls):
        """Reset accessor state for test suite initialization only."""
        cls._accessed = False
        cls._access_log = []

    @classmethod
    def get_holdout_data(cls, df_merged, wave_cols):
        """
        Guarded accessor for the 37 reference holdout samples.
        Raises RuntimeError if called more than once during execution.
        """
        if cls._accessed:
            msg = "CRITICAL SAFETY VIOLATION: Reference holdout dataset (37 samples) was accessed MORE THAN ONCE!"
            cls._access_log.append({"status": "REJECTED_REPEATED_ACCESS", "message": msg})
            raise RuntimeError(msg)

        cls._accessed = True
        test_df = df_merged[df_merged['split'] == 'test'].reset_index(drop=True)
        if len(test_df) == 0:
            raise ValueError("No test split samples found in df_merged!")

        X_test = test_df[wave_cols].values.astype(np.float64)
        y_test = test_df['Dry matter'].values.astype(np.float64)
        test_ids = test_df['Mush_str'].values

        log_entry = {
            "status": "GRANTED_SINGLE_ACCESS",
            "sample_count": len(X_test),
            "holdout_dmc_mean": float(np.mean(y_test)),
            "holdout_dmc_sd": float(np.std(y_test, ddof=1))
        }
        cls._access_log.append(log_entry)
        print(f"[GUARDED HOLDOUT ACCESSOR] Single access GRANTED for {len(X_test)} reference holdout samples.")
        return X_test, y_test, test_ids
