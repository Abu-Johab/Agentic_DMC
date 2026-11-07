import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.ndimage import median_filter

# Global Font Setup for Arial Black
FONT_PATH = "C:/Windows/Fonts/ariblk.ttf"
if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
    except Exception:
        pass

def apply_arial_black_typography(fig, axes, title_size=16, label_size=12):
    """Enforces Arial Black typography across all figure components."""
    plt.rcParams["font.sans-serif"] = ["Arial", "Arial Black"]
    plt.rcParams["font.family"] = "sans-serif"
    
    if not isinstance(axes, (list, np.ndarray)):
        axes_list = [axes]
    else:
        axes_list = np.array(axes).flatten()
        
    for ax in axes_list:
        if ax is None:
            continue
        # Title
        title_text = ax.get_title()
        if title_text:
            ax.set_title(title_text, fontname="Arial", fontweight="black", fontsize=title_size)
            
        # X and Y labels
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontname="Arial", fontweight="black", fontsize=label_size)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontname="Arial", fontweight="black", fontsize=label_size)
            
        # Ticks
        ax.tick_params(labelsize=label_size)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontname("Arial")
            t.set_fontweight("black")
            
        # Legend
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontname("Arial")
                t.set_fontweight("black")
                t.set_fontsize(label_size)

class SpatialTool:
    """
    Deterministic Spatial Visualization Engine:
    Executes per-pixel DMC inference, 3x3 spatial median filtering, pseudo-RGB composite (Bands 120, 60, 20),
    out-of-domain checking, pixel histogram distribution analysis, and CSV spectrum sanity check.
    """
    
    @staticmethod
    def generate_spatial_maps(cube, mask, pipeline_pkg, sample_id="M160", output_dir="agentic_dmc/outputs/spatial_maps", num_bands=204, true_dmc=None):
        """
        cube: numpy array (H, W, Bands) or (Bands, H, W)
        mask: boolean array (H, W)
        pipeline_pkg: dict with {"preprocessor": PreprocessingTool, "model": model_object}
        true_dmc: optional measured DMC float label
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if cube is None or cube.ndim != 3:
            return None, {"error": "Cube must be 3D"}, "REJECTED"
            
        if cube.shape[2] == num_bands:
            cube_hw = cube.copy()
        elif cube.shape[0] == num_bands:
            cube_hw = np.moveaxis(cube, 0, -1)
        else:
            if cube.shape[0] < cube.shape[2] and cube.shape[2] > 100:
                cube_hw = np.moveaxis(cube, 0, -1)
            else:
                cube_hw = cube.copy()
            
        H, W, B = cube_hw.shape
        if mask.shape != (H, W):
            return None, {"error": f"Mask shape {mask.shape} mismatch with cube {(H, W)}"}, "REJECTED"

        # 1. Pseudo-RGB Composite Synthesis (Band 120: Red, Band 60: Green, Band 20: Blue)
        r = cube_hw[:, :, min(120, B-1)].astype(np.float32)
        g = cube_hw[:, :, min(60, B-1)].astype(np.float32)
        b = cube_hw[:, :, min(20, B-1)].astype(np.float32)
        
        def norm_channel(c):
            c_min, c_max = c.min(), c.max()
            if c_max > c_min:
                return np.clip((c - c_min) / (c_max - c_min + 1e-8), 0, 1)
            return np.zeros_like(c)
            
        pseudo_rgb = np.dstack([norm_channel(r), norm_channel(g), norm_channel(b)])

        # 2. Pixel-wise Preprocessing and Model Batch Inference
        preproc = pipeline_pkg["preprocessor"]
        model = pipeline_pkg["model"]
        
        dmc_map_raw = np.full((H, W), np.nan, dtype=np.float32)
        validity_map = np.zeros((H, W), dtype=np.uint8) # 0: BG, 1: Valid ROI, 2: Out-of-Domain ROI
        
        roi_indices = np.where(mask)
        roi_pixels_raw = cube_hw[roi_indices] # shape: (N_roi, B)
        
        if len(roi_pixels_raw) > 0:
            roi_pixels_proc = preproc.transform(roi_pixels_raw)
            
            batch_size = 2048
            predictions = []
            for start_idx in range(0, len(roi_pixels_proc), batch_size):
                end_idx = min(start_idx + batch_size, len(roi_pixels_proc))
                batch = roi_pixels_proc[start_idx:end_idx]
                p = model.predict(batch)
                predictions.append(p)
            roi_preds = np.concatenate(predictions)
            roi_preds = np.clip(roi_preds, 0.0, 1.0)
            dmc_map_raw[roi_indices] = roi_preds
            validity_map[roi_indices] = 1
            
            # Out-of-domain check via Mahalanobis distance if covariance matrix exists
            if "X_tr_mean" in pipeline_pkg and "X_tr_cov_inv" in pipeline_pkg:
                mean_vec = pipeline_pkg["X_tr_mean"]
                cov_inv = pipeline_pkg["X_tr_cov_inv"]
                diff = roi_pixels_proc - mean_vec
                mahal = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
                out_of_domain = mahal > (np.mean(mahal) + 3.0 * np.std(mahal))
                ood_sub_indices = (roi_indices[0][out_of_domain], roi_indices[1][out_of_domain])
                validity_map[ood_sub_indices] = 2

        # 3. 3x3 Spatial Median Filtering over non-background predictions
        dmc_map_filled = np.nan_to_num(dmc_map_raw, nan=0.0)
        dmc_map_filtered = median_filter(dmc_map_filled, size=3)
        dmc_map_filtered[~mask] = np.nan
        
        mean_pixel_dmc = float(np.nanmean(dmc_map_filtered)) if len(roi_pixels_raw) > 0 else 0.0
        
        # 4. Spatial Pixel-Mean vs CSV Spectrum Sanity Check
        mean_spectrum_raw = roi_pixels_raw.mean(axis=0).reshape(1, -1) if len(roi_pixels_raw) > 0 else np.zeros((1, B))
        mean_spec_proc = preproc.transform(mean_spectrum_raw)
        csv_spectrum_pred_dmc = float(model.predict(mean_spec_proc).ravel()[0]) if len(roi_pixels_raw) > 0 else 0.0
        sanity_diff = float(abs(mean_pixel_dmc - csv_spectrum_pred_dmc))

        # 5. Render 4-Panel Visualization Figure with Strict Arial Black Typography
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        
        # Panel 1: Pseudo-RGB Composite
        axes[0].imshow(pseudo_rgb)
        title_p1 = f"{sample_id} - Pseudo-RGB"
        if true_dmc is not None:
            title_p1 += f"\nTrue DMC: {true_dmc:.4f}"
        axes[0].set_title(title_p1)
        axes[0].axis("off")
        
        # Panel 2: Spatial DMC Map
        im = axes[1].imshow(dmc_map_filtered, cmap="viridis")
        axes[1].set_title(f"{sample_id} - Spatial DMC Map\nPixel Mean: {mean_pixel_dmc:.4f}")
        axes[1].axis("off")
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label("Dry Matter Content (DMC)", fontname="Arial", fontweight="black", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        for t in cbar.ax.get_yticklabels():
            t.set_fontname("Arial")
            t.set_fontweight("black")
        
        # Panel 3: Pixel DMC Distribution Histogram
        valid_vals = dmc_map_filtered[~np.isnan(dmc_map_filtered)]
        if len(valid_vals) > 0:
            axes[2].hist(valid_vals, bins=25, color="teal", alpha=0.7, edgecolor="black")
            if true_dmc is not None:
                axes[2].axvline(true_dmc, color="red", linestyle="--", linewidth=2, label="Measured")
            axes[2].axvline(mean_pixel_dmc, color="blue", linestyle="-", linewidth=2, label="Pixel Mean")
            axes[2].legend(loc="upper right")
        axes[2].set_title(f"{sample_id} - Pixel DMC Distribution")
        axes[2].set_xlabel("DMC Fraction")
        axes[2].set_ylabel("Pixel Count")
        
        # Panel 4: Validity Layer Map
        axes[3].imshow(validity_map, cmap="Accent")
        axes[3].set_title(f"{sample_id} - Validity Layer\n(0:BG, 1:Valid, 2:OOD)")
        axes[3].axis("off")

        apply_arial_black_typography(fig, axes, title_size=16, label_size=12)
        
        fig_path = os.path.join(output_dir, f"{sample_id}_spatial_dmc_map.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save raw compressed arrays
        np.savez_compressed(
            os.path.join(output_dir, f"{sample_id}_maps.npz"),
            dmc_map=dmc_map_filtered,
            validity_map=validity_map,
            mask=mask
        )

        qc_stats = {
            "sample_id": sample_id,
            "mean_spatial_dmc": round(mean_pixel_dmc, 4),
            "csv_spectrum_pred_dmc": round(csv_spectrum_pred_dmc, 4),
            "sanity_diff": round(sanity_diff, 4),
            "std_spatial_dmc": float(np.std(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "min_spatial_dmc": float(np.min(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "max_spatial_dmc": float(np.max(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "figure_path": fig_path
        }

        status = "PASS" if sanity_diff < 0.05 else "PASS_WITH_WARNING"
        return dmc_map_filtered, qc_stats, status

