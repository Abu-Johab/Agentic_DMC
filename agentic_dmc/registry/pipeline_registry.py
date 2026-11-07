import os
import pickle
import json

class PipelineRegistry:
    """
    Serializes and registers frozen pipeline packages.
    Package contains:
    - Preprocessing transformer (fitted with MSC reference & scaler on calibration set)
    - Model object
    - Wavelength vector (204 bands)
    - Calibration set Mahalanobis mean vector & inverse covariance matrix
    - Target definition & scaler parameters
    - Configuration & evaluation summary
    """
    
    REGISTRY_DIR = "agentic_dmc/registry"

    @classmethod
    def register_pipeline(cls, preprocessor, model, wavelengths, X_tr_mean, X_tr_cov_inv, eval_summary, filename="frozen_pipeline.pkl"):
        os.makedirs(cls.REGISTRY_DIR, exist_ok=True)
        filepath = os.path.join(cls.REGISTRY_DIR, filename)
        
        pkg = {
            "preprocessor": preprocessor,
            "model": model,
            "wavelengths": wavelengths,
            "X_tr_mean": X_tr_mean,
            "X_tr_cov_inv": X_tr_cov_inv,
            "eval_summary": eval_summary
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(pkg, f)
            
        # Also save JSON metadata summary
        json_path = os.path.join(cls.REGISTRY_DIR, filename.replace(".pkl", "_metadata.json"))
        meta = {
            "num_bands": len(wavelengths),
            "preprocessor_method": preprocessor.method,
            "model_type": str(model.__class__.__name__),
            "eval_summary": eval_summary
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        print(f"Registered frozen pipeline package -> {filepath}")
        return filepath

    @classmethod
    def load_pipeline(cls, filename="frozen_pipeline.pkl"):
        filepath = os.path.join(cls.REGISTRY_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file {filepath} not found!")
            
        with open(filepath, "rb") as f:
            pkg = pickle.load(f)
        return pkg
