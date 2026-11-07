import numpy as np

class ExtractionTool:
    """
    Deterministic spectral extraction tool:
    Accepts an HSI cube and boolean ROI mask, cleans invalid/saturated pixels,
    and computes the representative spectrum.
    """
    
    @staticmethod
    def extract_spectrum(cube, mask, sample_id="unknown", num_bands=204):
        """
        cube: numpy array of shape (H, W, Bands) or (Bands, H, W)
        mask: boolean array of shape (H, W)
        Returns:
            extracted_spectrum: 1D numpy array of shape (Bands,)
            qc_stats: dict of quality metrics
            status: PASS, PASS_WITH_WARNING, or REJECTED
        """
        if cube.ndim != 3:
            return None, {"error": "Cube must be 3D"}, "REJECTED"
            
        # Strict band count check
        if cube.shape[2] == num_bands:
            cube_hw = cube.copy()
        elif cube.shape[0] == num_bands:
            cube_hw = np.moveaxis(cube, 0, -1)
        else:
            return None, {"error": f"Wavelength band count mismatch: expected {num_bands} bands, but cube has dimensions {cube.shape}"}, "REJECTED"
            
        H, W, B = cube_hw.shape
        if mask.shape != (H, W):
            return None, {"error": f"Mask shape {mask.shape} mismatch with cube spatial dimensions {(H, W)}"}, "REJECTED"

        # Extract spatial pixels inside mask
        roi_pixels = cube_hw[mask] # shape: (N_roi, Bands)
        if len(roi_pixels) == 0:
            return None, {"error": "ROI mask has 0 foreground pixels"}, "REJECTED"

        # Check invalid values (NaN, Inf)
        invalid_mask = np.isnan(roi_pixels) | np.isinf(roi_pixels)
        invalid_count = int(np.sum(invalid_mask))
        
        # Check saturated pixels
        max_val = np.max(roi_pixels[~invalid_mask]) if np.any(~invalid_mask) else 1.0
        saturated_pixels = np.any(roi_pixels >= (0.999 * max_val if max_val > 1.0 else 0.999), axis=1)
        saturated_count = int(np.sum(saturated_pixels))
        
        # Filter valid pixels
        valid_pixel_mask = ~np.any(invalid_mask, axis=1) & ~saturated_pixels
        valid_roi_pixels = roi_pixels[valid_pixel_mask]
        
        if len(valid_roi_pixels) == 0:
            return None, {"error": "No valid pixels remaining after quality filtering"}, "REJECTED"

        # Representative spectrum: Mean spectrum over valid ROI pixels
        mean_spectrum = np.mean(valid_roi_pixels, axis=0)
        std_spectrum = np.std(valid_roi_pixels, axis=0)

        qc_stats = {
            "sample_id": sample_id,
            "total_roi_pixels": len(roi_pixels),
            "valid_roi_pixels": len(valid_roi_pixels),
            "invalid_pixel_count": invalid_count,
            "saturated_pixel_count": saturated_count,
            "spectrum_min": float(np.min(mean_spectrum)),
            "spectrum_max": float(np.max(mean_spectrum)),
            "spectrum_mean": float(np.mean(mean_spectrum)),
            "spectral_snr_estimate": float(np.mean(mean_spectrum) / (np.mean(std_spectrum) + 1e-8))
        }

        # Status rules
        valid_ratio = len(valid_roi_pixels) / len(roi_pixels)
        if valid_ratio < 0.50:
            status = "REJECTED"
            qc_stats["warning"] = f"Too many invalid/saturated pixels (valid ratio {valid_ratio:.4f})"
        elif valid_ratio < 0.85:
            status = "PASS_WITH_WARNING"
            qc_stats["warning"] = f"Some saturated or invalid pixels filtered (valid ratio {valid_ratio:.4f})"
        else:
            status = "PASS"

        return mean_spectrum, qc_stats, status
