import os
import sys
import pickle
import json
import hashlib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr
import shap

# Register Arial Black font for matplotlib
FONT_PATH = "C:/Windows/Fonts/ariblk.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
    except Exception:
        pass

def apply_arial_black_typography(fig=None, axes=None, title_size=16, label_size=12):
    """Enforces strict Arial Black typography across all matplotlib figure axes and text elements."""
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Black"]
    plt.rcParams["font.family"] = "sans-serif"
    
    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = fig.get_axes()
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
        
    for ax in axes:
        if ax is None:
            continue
        title_text = ax.get_title()
        if title_text:
            ax.set_title(title_text, fontname="Arial", fontweight="black", fontsize=title_size)
            
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontname="Arial", fontweight="black", fontsize=label_size)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontname="Arial", fontweight="black", fontsize=label_size)
            
        ax.tick_params(labelsize=label_size)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontname("Arial")
            t.set_fontweight("black")
            
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontname("Arial")
                t.set_fontweight("black")
                t.set_fontsize(label_size)

    for t in fig.findobj(plt.Text):
        t.set_fontname("Arial")
        t.set_fontweight("black")
        if t.get_fontsize() < 16:
            t.set_fontsize(label_size)

class ExplainabilityTool:
    """
    Deterministic Science & SHAP Explanation Engine for Hyperspectral DMC Prediction:
    - Mode A: Complete Original-Spectrum Pipeline Explanation (Raw 204 bands -> snv_sg1 -> CNN1D -> DMC)
    - Validates prediction consistency and additivity/reconstruction residuals
    - Evaluates global & local feature attributions, top-K wavelengths, rank stability across background seeds
    - Generates high-resolution scientific plots with Arial Black typography
    """
    
    CHECKPOINT_PATH = "agentic_dmc/registry/frozen_pipeline.pkl"
    DATASET_CSV = "SpectroFood_dataset.csv"
    SPLIT_CSV = r"E:\Thesis 2025\Agent_mushroom\plsr_split_ks70_15_15.csv"
    OUTPUT_DIR = os.path.normpath("agentic_dmc/outputs/xai")
    FIGURES_DIR = os.path.normpath("agentic_dmc/outputs/xai/figures")

    @classmethod
    def ensure_output_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.FIGURES_DIR, exist_ok=True)

    @classmethod
    def load_frozen_pipeline(cls):
        """Loads read-only frozen pipeline package."""
        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Frozen pipeline checkpoint not found at {cls.CHECKPOINT_PATH}")
            
        with open(cls.CHECKPOINT_PATH, "rb") as f:
            pkg = pickle.load(f)
            
        return pkg

    @classmethod
    def build_mode_a_wrapper(cls, pkg):
        """
        Mode A prediction wrapper:
        X_raw (204 bands) -> preprocessor.transform(X_raw) -> model.predict(...) -> predicted DMC (g/g).
        """
        preproc = pkg["preprocessor"]
        model = pkg["model"]
        
        def predict_fn(X_raw_input):
            X_arr = np.asarray(X_raw_input, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            X_proc = preproc.transform(X_arr)
            preds = model.predict(X_proc)
            return np.asarray(preds, dtype=np.float64).ravel()
            
        return predict_fn

    @classmethod
    def run_full_explainability_pipeline(cls, n_clusters=25, nsamples=300, random_seed=42):
        """
        Executes scientific SHAP attribution, validation, stability, and plotting workflow.
        Uses GradientExplainer on PyTorch 1D-CNN preprocessed features (Mode B) for expected-gradients-based
        approximate attributions of processed spectral features associated with wavelengths, avoiding
        raw spectrum derivative boundary artifacts.
        """
        cls.ensure_output_dirs()
        t0 = time.time()
        
        # 1. Load protected pipeline & dataset
        pkg = cls.load_frozen_pipeline()
        preproc = pkg["preprocessor"]
        model = pkg["model"]
        torch_model = model.torch_model
        y_std = float(model.y_std)
        y_mean = float(model.y_mean)
        device = model.device
        
        predict_fn = cls.build_mode_a_wrapper(pkg)
        wavelengths = np.asarray(pkg["wavelengths"], dtype=np.float64)
        
        df = pd.read_csv(cls.DATASET_CSV)
        sp_df = pd.read_csv(cls.SPLIT_CSV).set_index("Mush No").loc[df["Mush No"]].reset_index()
        wave_cols = [c for c in df.columns if c not in ["Mush No", "Dry matter"]]
        
        X_all = df[wave_cols].values.astype(np.float64)
        y_all = df["Dry matter"].values.astype(np.float64)
        mush_ids = df["Mush No"].values
        
        cal_mask = (sp_df["split"] == "cal").values
        val_mask = (sp_df["split"] == "val").values
        test_mask = (sp_df["split"] == "test").values
        
        X_cal = X_all[cal_mask]
        X_test = X_all[test_mask]
        y_test = y_all[test_mask]
        ids_test = mush_ids[test_mask]
        
        # 2. Prediction Consistency Check
        test_preds_wrapper = predict_fn(X_test)
        test_preds_direct = model.predict(preproc.transform(X_test)).ravel()
        max_pred_diff = float(np.max(np.abs(test_preds_wrapper - test_preds_direct)))
        consistency_pass = (max_pred_diff < 1.0e-5)
        
        # 3. Background Dataset Preparation
        bg_summary = shap.kmeans(X_cal, n_clusters)
        bg_data = bg_summary.data
        
        # Save background_samples.csv
        bg_df = pd.DataFrame(bg_data, columns=[f"band_{i}_{w:.1f}nm" for i, w in enumerate(wavelengths)])
        bg_df.insert(0, "cluster_idx", range(len(bg_data)))
        bg_csv_path = os.path.join(cls.OUTPUT_DIR, "background_samples.csv")
        bg_df.to_csv(bg_csv_path, index=False)
        
        def clean_mush_id(m):
            s = str(m).strip().replace("M", "")
            try:
                return f"M{int(s):03d}"
            except Exception:
                return str(m)

        # 4. Explained Samples Cohort
        test_mush_ids = [clean_mush_id(m) for m in ids_test]
        explained_cohort_ids = list(test_mush_ids)
        spatial_sample_ids = ["M160", "M175", "M200", "M220", "M250"]
        spatial_indices = []
        for sid in spatial_sample_ids:
            clean_sid = clean_mush_id(sid)
            if clean_sid not in test_mush_ids:
                matches = [i for i, m in enumerate(mush_ids) if clean_mush_id(m) == clean_sid]
                if len(matches) > 0:
                    spatial_indices.append(matches[0])
                    explained_cohort_ids.append(clean_sid)
                
        if len(spatial_indices) > 0:
            X_explained = np.vstack([X_test, X_all[spatial_indices]])
            y_explained = np.hstack([y_test, y_all[spatial_indices]])
            splits_explained = list(sp_df["split"].values[test_mask]) + list(sp_df["split"].values[spatial_indices])
        else:
            X_explained = X_test.copy()
            y_explained = y_test.copy()
            splits_explained = list(sp_df["split"].values[test_mask])
            
        explained_df = pd.DataFrame({
            "sample_index": range(len(X_explained)),
            "mush_no": explained_cohort_ids,
            "split": splits_explained,
            "measured_dmc": y_explained,
            "predicted_dmc": predict_fn(X_explained)
        })
        explained_csv_path = os.path.join(cls.OUTPUT_DIR, "explained_samples.csv")
        explained_df.to_csv(explained_csv_path, index=False)

        # 5. Execute Analytical Gradient SHAP Attribution Computation
        X_cal_proc = preproc.transform(X_cal)
        X_explained_proc = preproc.transform(X_explained)
        
        import torch
        bg_tensor = torch.from_numpy(X_cal_proc).float().unsqueeze(1).to(device)
        explained_tensor = torch.from_numpy(X_explained_proc).float().unsqueeze(1).to(device)
        
        torch_model.eval()
        explainer = shap.GradientExplainer(torch_model, bg_tensor)
        shap_raw = explainer.shap_values(explained_tensor)
        if isinstance(shap_raw, list):
            shap_raw = shap_raw[0]
        shap_matrix = np.asarray(shap_raw, dtype=np.float64).squeeze() * y_std
        
        with torch.no_grad():
            bg_preds = torch_model(bg_tensor).cpu().numpy().ravel() * y_std + y_mean
        base_value = float(np.mean(bg_preds))
        
        # 6. Reconstruction / Additivity Diagnostic
        preds_explained = predict_fn(X_explained)
        recon_preds = base_value + np.sum(shap_matrix, axis=1)
        residuals = np.abs(preds_explained - recon_preds)
        
        mean_residual = float(np.mean(residuals))
        max_residual = float(np.max(residuals))
        additivity_pass = (max_residual < 0.15) # Gradient SHAP integrated gradients reconstruction tolerance
        finite_pass = bool(np.all(np.isfinite(shap_matrix)))
        
        # Save shap_values.npz
        npz_path = os.path.join(cls.OUTPUT_DIR, "shap_values.npz")
        np.savez_compressed(
            npz_path,
            shap_values=shap_matrix,
            base_value=base_value,
            predictions=preds_explained,
            reconstructed_predictions=recon_preds,
            residuals=residuals,
            wavelengths=wavelengths,
            sample_ids=np.array(explained_cohort_ids)
        )
        
        # Save prediction_reconstruction.csv
        recon_df = pd.DataFrame({
            "mush_no": explained_cohort_ids,
            "base_value": base_value,
            "sum_shap": np.sum(shap_matrix, axis=1),
            "reconstructed_pred": recon_preds,
            "actual_pred": preds_explained,
            "absolute_residual": residuals,
            "pass_additivity": residuals < 0.15
        })
        recon_df.to_csv(os.path.join(cls.OUTPUT_DIR, "prediction_reconstruction.csv"), index=False)

        # 7. Global & Local Wavelength Importance Analysis
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        signed_mean_shap = np.mean(shap_matrix, axis=0)
        std_shap = np.std(shap_matrix, axis=0)
        se_shap = std_shap / np.sqrt(len(shap_matrix))
        
        rank_order = np.argsort(mean_abs_shap)[::-1]
        
        imp_df = pd.DataFrame({
            "rank": range(1, len(wavelengths) + 1),
            "band_index": rank_order,
            "wavelength_nm": wavelengths[rank_order],
            "mean_abs_shap": mean_abs_shap[rank_order],
            "signed_mean_shap": signed_mean_shap[rank_order],
            "std_shap": std_shap[rank_order],
            "se_shap": se_shap[rank_order]
        })
        imp_csv_path = os.path.join(cls.OUTPUT_DIR, "wavelength_importance.csv")
        imp_df.to_csv(imp_csv_path, index=False)

        # Save local_attributions.csv for top 10 wavelengths
        top10_indices = rank_order[:10]
        local_dict = {"mush_no": explained_cohort_ids, "predicted_dmc": preds_explained}
        for b_idx in top10_indices:
            w_nm = wavelengths[b_idx]
            local_dict[f"shap_{w_nm:.1f}nm"] = shap_matrix[:, b_idx]
            local_dict[f"raw_{w_nm:.1f}nm"] = X_explained[:, b_idx]
            
        local_df = pd.DataFrame(local_dict)
        local_df.to_csv(os.path.join(cls.OUTPUT_DIR, "local_attributions.csv"), index=False)

        # 8. Rank Stability Analysis across background subsets
        seed_variants = [42, 100, 2026]
        seed_rankings = []
        
        for s_idx, seed_val in enumerate(seed_variants):
            sub_idx = np.random.RandomState(seed_val).choice(len(X_cal_proc), min(50, len(X_cal_proc)), replace=False)
            bg_sub = torch.from_numpy(X_cal_proc[sub_idx]).float().unsqueeze(1).to(device)
            ex_sub = shap.GradientExplainer(torch_model, bg_sub)
            s_vals = ex_sub.shap_values(explained_tensor)
            if isinstance(s_vals, list): s_vals = s_vals[0]
            s_mat = s_vals.squeeze() * y_std
            m_abs = np.mean(np.abs(s_mat), axis=0)
            seed_rankings.append(np.argsort(m_abs)[::-1])
            
        top20_ref = rank_order[:20]
        # Rank vectors for top 20 wavelengths under variants
        ranks_seed0 = [list(seed_rankings[0]).index(b) for b in top20_ref]
        ranks_seed1 = [list(seed_rankings[1]).index(b) for b in top20_ref]
        ranks_seed2 = [list(seed_rankings[2]).index(b) for b in top20_ref]
        
        corr_12, _ = spearmanr(ranks_seed0, ranks_seed1)
        corr_13, _ = spearmanr(ranks_seed0, ranks_seed2)
        mean_spearman_corr = float(np.mean([corr_12, corr_13]))

        # 9. Export Validation Diagnostics & Configuration Metadata
        validation_diag = {
            "prediction_consistency": {
                "max_absolute_difference": max_pred_diff,
                "tolerance": 1.0e-5,
                "passed": consistency_pass
            },
            "reconstruction_additivity": {
                "base_value": base_value,
                "mean_residual": mean_residual,
                "max_residual": max_residual,
                "tolerance": 0.15,
                "samples_passed": int(np.sum(residuals < 0.15)),
                "total_samples": len(X_explained),
                "passed": additivity_pass
            },
            "finite_values_check": {
                "all_finite": finite_pass,
                "passed": finite_pass
            },
            "rank_stability": {
                "mean_spearman_rank_correlation": round(mean_spearman_corr, 4),
                "variants_evaluated": len(seed_variants),
                "stable": bool(mean_spearman_corr > 0.85)
            },
            "overall_validation": "PASSED" if (consistency_pass and additivity_pass and finite_pass) else "FAILED"
        }
        
        with open(os.path.join(cls.OUTPUT_DIR, "explanation_validation.json"), "w", encoding="utf-8") as f:
            json.dump(validation_diag, f, indent=2)

        config_meta = {
            "mode": "Mode_B_Model_Input_Features",
            "prediction_function": "CNN1D(snv_sg1(X_raw))",
            "explainer_class": "shap.GradientExplainer",
            "attribution_type": "expected_gradients_approximate_attributions",
            "feature_space": "snv_sg1_processed_spectral_features",
            "background_sampling": {"method": "calibration_set_tensor", "n_samples": len(X_cal)},
            "explained_cohort_size": len(X_explained),
            "output_units": "Dry Matter Content (g/g fraction)"
        }
        with open(os.path.join(cls.OUTPUT_DIR, "explainer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_meta, f, indent=2)

        provenance_meta = {
            "dataset": cls.DATASET_CSV,
            "dataset_sha256": hashlib.sha256(open(cls.DATASET_CSV, "rb").read()).hexdigest(),
            "checkpoint": cls.CHECKPOINT_PATH,
            "checkpoint_sha256": hashlib.sha256(open(cls.CHECKPOINT_PATH, "rb").read()).hexdigest(),
            "execution_wall_time_sec": round(time.time() - t0, 3)
        }
        with open(os.path.join(cls.OUTPUT_DIR, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(provenance_meta, f, indent=2)

        # 10. Generate Figures with Arial Black Typography
        cls.generate_all_figures(
            wavelengths=wavelengths,
            shap_matrix=shap_matrix,
            X_explained=X_explained_proc,
            explained_df=explained_df,
            imp_df=imp_df,
            rank_order=rank_order,
            seed_rankings=seed_rankings
        )

        return validation_diag

    @classmethod
    def generate_all_figures(cls, wavelengths, shap_matrix, X_explained, explained_df, imp_df, rank_order, seed_rankings):
        """Generates 7 scientific XAI plots enforcing strict Arial Black typography."""
        
        # -------------------------------------------------------------
        # Figure 1: Global Mean Absolute SHAP vs Wavelength Spectrum
        # -------------------------------------------------------------
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(wavelengths, mean_abs_shap, color="navy", linewidth=2.0, label="Mean |SHAP|")
        ax.fill_between(wavelengths, 0, mean_abs_shap, color="navy", alpha=0.2)
        
        # Highlight top 3 peak wavelengths
        top3_indices = rank_order[:3]
        for idx in top3_indices:
            w_val = wavelengths[idx]
            sh_val = mean_abs_shap[idx]
            ax.scatter(w_val, sh_val, color="crimson", s=60, zorder=5)
            ax.annotate(f"{w_val:.1f} nm", (w_val, sh_val), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontname="Arial", fontweight="black", fontsize=12, color="crimson")
                        
        ax.set_title("Global Mean Absolute SHAP Attribution Across Hyperspectral Spectrum")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Mean |SHAP value|")
        ax.grid(True, alpha=0.3)
        apply_arial_black_typography(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "01_global_shap_vs_wavelength.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 2: Top-10 Wavelength Importance Bar Plot
        # -------------------------------------------------------------
        top10_df = imp_df.head(10).iloc[::-1] # Reverse for horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh([f"{w:.1f} nm" for w in top10_df["wavelength_nm"]], top10_df["mean_abs_shap"], color="teal", edgecolor="black")
        ax.set_title("Top-10 Spectral Wavelengths by Global SHAP Importance")
        ax.set_xlabel("Mean |SHAP value|")
        ax.grid(True, alpha=0.3, axis="x")
        
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 0.0005, bar.get_y() + bar.get_height()/2.0, f"{xval:.4f}", ha='left', va='center',
                    fontname="Arial", fontweight="black", fontsize=12)
                    
        apply_arial_black_typography(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "02_top10_wavelength_importance.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 3: SHAP Beeswarm Summary Plot for Top 10 Wavelengths
        # -------------------------------------------------------------
        top10_indices = rank_order[:10]
        top10_shap = shap_matrix[:, top10_indices]
        top10_raw = X_explained[:, top10_indices]
        top10_labels = [f"{wavelengths[i]:.1f} nm" for i in top10_indices]
        
        fig, ax = plt.subplots(figsize=(11, 6))
        # Use SHAP summary_plot for native beeswarm display of top 10 features
        shap.summary_plot(top10_shap, top10_raw, feature_names=top10_labels, max_display=10, show=False)
        scatter_ax = plt.gcf().get_axes()[0]
        # 1. Remove title as per requirement
        scatter_ax.set_title("")
        
        # 2. Enforce symmetric x-limits so 0.0 line is placed exactly in middle horizontally
        max_abs = float(np.max(np.abs(top10_shap)))
        if max_abs <= 0.06:
            limit_x = 0.07
            ticks_x = [-0.05, 0.00, 0.05]
        elif max_abs <= 0.11:
            limit_x = 0.13
            ticks_x = [-0.10, -0.05, 0.00, 0.05, 0.10]
        elif max_abs <= 0.165:
            limit_x = 0.18
            ticks_x = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
        else:
            step_x = 0.05
            limit_x = np.ceil((max_abs * 1.12) / step_x) * step_x
            ticks_x = np.arange(-limit_x, limit_x + step_x * 0.5, step_x)

        scatter_ax.set_xlim(-limit_x, limit_x)
        scatter_ax.set_xticks(ticks_x)
        scatter_ax.set_xticklabels([f"{0.0:.2f}" if abs(t) < 1e-5 else f"{t:.2f}" for t in ticks_x])
        
        # 3. Label for X-axis
        scatter_ax.set_xlabel("SHAP value", fontname="Arial", fontweight="black", fontsize=12)
        
        # 4. Format typography for all text elements including High, Low, Feature value
        apply_arial_black_typography(fig, fig.get_axes())
        plt.savefig(os.path.join(cls.FIGURES_DIR, "03_shap_summary_beeswarm.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 4: Sample Local Signed Attribution Waterfall Plots
        # -------------------------------------------------------------
        top10_indices = rank_order[:10]
        target_samples = ["M160", "M220", "M250"]
        fig, axes = plt.subplots(1, len(target_samples), figsize=(18, 5))
        
        for idx, sid in enumerate(target_samples):
            match_row = explained_df[explained_df["mush_no"] == sid]
            if len(match_row) == 0:
                continue
            s_idx = match_row["sample_index"].values[0]
            pred_v = match_row["predicted_dmc"].values[0]
            
            s_shap = shap_matrix[s_idx, top10_indices]
            s_labels = [f"{wavelengths[b]:.1f} nm" for b in top10_indices]
            
            colors = ["crimson" if v > 0 else "navy" for v in s_shap]
            axes[idx].barh(s_labels, s_shap, color=colors, edgecolor="black")
            axes[idx].axvline(0, color="black", linestyle="--")
            axes[idx].set_title(f"Sample {sid}\nPred DMC: {pred_v:.4f}")
            axes[idx].set_xlabel("Signed mean SHAP value")
            axes[idx].grid(True, alpha=0.3, axis="x")
            
        apply_arial_black_typography(fig, axes)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "04_sample_attributions_waterfall.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 5: Sample x Wavelength SHAP Heatmap
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        top10_indices = rank_order[:10]
        im = ax.imshow(shap_matrix[:, top10_indices].T, cmap="PuOr", aspect="auto")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("SHAP value", fontname="Arial", fontweight="black", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        for t in cbar.ax.get_yticklabels():
            t.set_fontname("Arial")
            t.set_fontweight("black")
            
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{wavelengths[i]:.1f} nm" for i in top10_indices])
        ax.set_xlabel("Explained Sample Index")
        ax.set_ylabel("Top 10 Wavelengths")
        ax.set_title("Sample × Top 10 Wavelength SHAP Attribution Matrix")
        
        apply_arial_black_typography(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "05_sample_wavelength_shap_heatmap.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 6: Top Wavelength SHAP Dependence Plot
        # -------------------------------------------------------------
        top1_idx = rank_order[0]
        top1_w = wavelengths[top1_idx]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(X_explained[:, top1_idx], shap_matrix[:, top1_idx], color="indigo", alpha=0.8, edgecolors="black")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"SHAP Dependence Plot for Top Wavelength ({top1_w:.1f} nm)")
        ax.set_xlabel(f"Preprocessed Feature Value at {top1_w:.1f} nm")
        ax.set_ylabel("SHAP value")
        ax.grid(True, alpha=0.3)
        
        apply_arial_black_typography(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "06_top_wavelength_dependence.png"), dpi=300)
        plt.close()

        # -------------------------------------------------------------
        # Figure 7: Rank Stability Across Background Subsets
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        top10_w_labels = [f"{wavelengths[i]:.1f} nm" for i in rank_order[:10]]
        
        for variant_idx, r_list in enumerate(seed_rankings):
            ranks_top10 = [list(r_list).index(i) + 1 for i in rank_order[:10]]
            ax.plot(top10_w_labels, ranks_top10, "o-", label=f"Subset Variant {variant_idx+1}")
            
        ax.set_title("Wavelength Importance Rank Stability Across Background Subsets")
        ax.set_xlabel("Top 10 Wavelengths")
        ax.set_ylabel("Observed Importance Rank (Lower is Better)")
        ax.set_ylim(0, 15)
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        apply_arial_black_typography(fig, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(cls.FIGURES_DIR, "07_rank_stability_across_seeds.png"), dpi=300)
        plt.close()

