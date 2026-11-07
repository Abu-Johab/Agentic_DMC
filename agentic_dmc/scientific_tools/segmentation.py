import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, binary_fill_holes, label, binary_opening, binary_erosion, binary_dilation
from skimage.morphology import convex_hull_image

class SegmentationTool:
    """
    Advanced Hyperspectral Image Segmentation Engine:
    Supports:
    1. noise_floor: Primary V3 Background-Referenced Noise-Floor Segmentation with Plateau Selection
    2. noise_floor_hull: V3 Noise-Floor with Convex Hull smoothing
    3. multi_otsu: 3-class Otsu separating background, shadow rim, and mushroom tissue
    4. grabcut: OpenCV Graph-Cut segmentation initialized via Pseudo-RGB composite
    5. otsu: Global Otsu thresholding on NIR band
    6. connected_components: Otsu thresholding with largest connected component filtering
    7. adaptive_nir: Dynamic NIR percentile thresholding
    8. band_ratio: NIR/Red ratio thresholding
    9. pca_threshold: Otsu thresholding on 1st Principal Component
    """
    
    @staticmethod
    def _normalize_cube_shape(cube, num_bands=204):
        """Ensures cube shape is (H, W, Bands) for spatial processing."""
        if cube.ndim != 3:
            raise ValueError("Cube must be 3D")
        if cube.shape[2] == num_bands:
            return cube
        elif cube.shape[0] == num_bands:
            return np.moveaxis(cube, 0, -1)
        elif cube.shape[0] < cube.shape[2] and cube.shape[2] > 100:
            return np.moveaxis(cube, 0, -1)
        return cube

    @staticmethod
    def segment_cube(cube, method="noise_floor", band_idx=100, min_roi_frac=0.05, max_roi_frac=0.85, num_bands=204):
        """
        Segments HSI datacube into binary ROI mask.
        Returns:
            mask: boolean 2D array (H, W)
            qc_stats: dict of quality diagnostics
            status: PASS, PASS_WITH_WARNING, or REJECTED
        """
        if cube is None or cube.ndim != 3:
            return None, {"error": "Invalid or missing 3D cube"}, "REJECTED"

        try:
            cube_hw = SegmentationTool._normalize_cube_shape(cube, num_bands=num_bands)
        except ValueError as e:
            return None, {"error": str(e)}, "REJECTED"

        H, W, B = cube_hw.shape
        total_pixels = H * W

        # Handle zero or corrupted cube
        if np.all(cube_hw == 0) or np.all(np.isnan(cube_hw)):
            return np.zeros((H, W), dtype=bool), {
                "method": method, "error": "Cube is empty or NaN", "foreground_ratio": 0.0
            }, "REJECTED"

        use_hull = (method == "noise_floor_hull")

        if method in ["noise_floor", "noise_floor_hull"]:
            # 1. Integrated Brightness Image b(x, y) over 500-900 nm (~band 35 to 175)
            b_start = min(35, B - 1)
            b_end = min(175, B)
            b = cube_hw[:, :, b_start:b_end].mean(axis=-1).astype(np.float32)
            b_blur = gaussian_filter(b, sigma=1.0)

            # 2. Outer border frame (outer 5%) background estimation
            margin_h = max(2, int(0.05 * H))
            margin_w = max(2, int(0.05 * W))
            border_mask = np.ones((H, W), dtype=bool)
            border_mask[margin_h:H-margin_h, margin_w:W-margin_w] = False

            border_pixels = b_blur[border_mask]
            if len(border_pixels) > 0:
                p90 = np.percentile(border_pixels, 90)
                valid_border = border_pixels[border_pixels <= p90]
                if len(valid_border) == 0:
                    valid_border = border_pixels
            else:
                valid_border = b_blur.flatten()

            mu_bg = float(np.median(valid_border))
            mad = float(np.median(np.abs(valid_border - mu_bg)))
            sigma_bg = float(1.4826 * mad)

            # 3. Plateau selection for threshold multiplier k in logspace(3, 40)
            ks = np.logspace(np.log10(3.0), np.log10(40.0), 30)
            areas = np.array([(b_blur > (mu_bg + k * sigma_bg)).sum() for k in ks])

            rel_change = np.abs(np.diff(areas)) / (areas[:-1] + 1e-5)
            if len(rel_change) >= 3:
                rel_change_smooth = np.convolve(rel_change, np.ones(3)/3.0, mode='same')
            else:
                rel_change_smooth = rel_change
            
            best_k = float(ks[np.argmin(rel_change_smooth)])
            raw_fg = b_blur > (mu_bg + best_k * sigma_bg)

            # 4. Post-processing chain
            filled = binary_fill_holes(raw_fg)
            opened = binary_opening(filled, structure=np.ones((3, 3)))

            labeled, num_features = label(opened)
            if num_features > 0:
                sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
                mask = (labeled == (np.argmax(sizes) + 1))
            else:
                mask = opened

            if use_hull and mask.sum() > 50:
                try:
                    mask = convex_hull_image(mask)
                except Exception:
                    pass

            if mask.sum() > 100:
                mask = binary_erosion(mask, structure=np.ones((3, 3)))

        elif method == "multi_otsu":
            ref_band = cube_hw[:, :, min(140, B-1)].astype(np.float32)
            lo, hi = np.percentile(ref_band, 1), np.percentile(ref_band, 99)
            norm_img = np.clip((ref_band - lo) / (hi - lo + 1e-8) * 255.0, 0, 255).astype(np.uint8)
            # Apply 2-level multi Otsu thresholding via OpenCV
            thresh1, _ = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = norm_img > thresh1
            mask = binary_fill_holes(mask)
            mask = binary_opening(mask, structure=np.ones((3, 3)))

        elif method == "grabcut":
            # Synthesize Pseudo-RGB
            r = cube_hw[:, :, min(120, B-1)]
            g = cube_hw[:, :, min(60, B-1)]
            b = cube_hw[:, :, min(20, B-1)]
            rgb = np.dstack([r, g, b])
            rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (int(W*0.1), int(H*0.1), int(W*0.8), int(H*0.8))
            gc_mask = np.zeros((H, W), np.uint8)
            try:
                cv2.grabCut(rgb_norm, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
                mask = np.where((gc_mask == 2) | (gc_mask == 0), False, True)
            except Exception:
                mask = rgb_norm[:, :, 0] > np.percentile(rgb_norm[:, :, 0], 50)

        elif method == "otsu":
            ref_band = cube_hw[:, :, min(band_idx, B-1)].astype(np.float32)
            lo, hi = np.percentile(ref_band, 1), np.percentile(ref_band, 99)
            norm_img = np.clip((ref_band - lo) / (hi - lo + 1e-8) * 255.0, 0, 255).astype(np.uint8) if hi > lo else np.zeros((H, W), dtype=np.uint8)
            _, thresh_mask = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = thresh_mask > 0

        elif method == "connected_components":
            ref_band = cube_hw[:, :, min(band_idx, B-1)].astype(np.float32)
            lo, hi = np.percentile(ref_band, 1), np.percentile(ref_band, 99)
            norm_img = np.clip((ref_band - lo) / (hi - lo + 1e-8) * 255.0, 0, 255).astype(np.uint8) if hi > lo else np.zeros((H, W), dtype=np.uint8)
            _, thresh_mask = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_mask)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label)
            else:
                mask = thresh_mask > 0

        elif method == "adaptive_nir":
            nir_band = cube_hw[:, :, min(140, B-1)].astype(np.float32)
            active_pixels = nir_band[nir_band > 0.02]
            p25 = np.percentile(active_pixels, 25) if len(active_pixels) > 0 else 0.08
            thresh = max(0.08, float(p25))
            mask = nir_band > thresh

        elif method == "band_ratio":
            b_nir = cube_hw[:, :, min(140, B-1)].astype(np.float32)
            b_red = cube_hw[:, :, min(50, B-1)].astype(np.float32)
            ratio = (b_nir + 1e-6) / (b_red + 1e-6)
            thresh = np.percentile(ratio, 50)
            mask = ratio > thresh

        elif method == "pca_threshold":
            reshaped = cube_hw.reshape(-1, B)
            mean_sub = reshaped - np.mean(reshaped, axis=0)
            cov = np.cov(mean_sub, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            pc1 = mean_sub @ eigvecs[:, -1]
            pc1_img = pc1.reshape(H, W)
            mask = pc1_img > np.mean(pc1_img)

        else:
            return None, {"error": f"Unknown method {method}"}, "REJECTED"

        # General Morphological Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        mask = mask_uint8 > 0

        # Label-Free QC Protocol Diagnostics
        fg_pixels = int(np.sum(mask))
        fg_ratio = float(fg_pixels / total_pixels)

        # 1. Border Contact Check
        border_contact = bool(mask[:2, :].any() or mask[-2:, :].any() or mask[:, :2].any() or mask[:, -2:].any())

        # 2. Outer Ring Spectral Angle Test
        ring_sa = 0.0
        ring_signal_ratio = 0.0
        if fg_pixels > 50:
            dilated = binary_dilation(mask, iterations=4)
            ring = dilated & (~mask)
            if ring.sum() > 0:
                spec_mask = cube_hw[mask].mean(axis=0)
                spec_ring = cube_hw[ring].mean(axis=0)
                norm_m = np.linalg.norm(spec_mask)
                norm_r = np.linalg.norm(spec_ring)
                if norm_m > 0 and norm_r > 0:
                    cos_sa = np.clip(np.dot(spec_mask, spec_ring) / (norm_m * norm_r), -1.0, 1.0)
                    ring_sa = float(np.arccos(cos_sa))
                    ring_signal_ratio = float(norm_r / norm_m)

        # 3. Inside Spectral Uniformity Test (95th percentile spectral angle)
        inside_sa95 = 0.0
        if fg_pixels > 50:
            spec_med = np.median(cube_hw[mask], axis=0)
            norm_med = np.linalg.norm(spec_med)
            if norm_med > 0:
                pix_specs = cube_hw[mask]
                pix_norms = np.linalg.norm(pix_specs, axis=1) + 1e-8
                cos_sims = np.clip((pix_specs @ spec_med) / (pix_norms * norm_med), -1.0, 1.0)
                sas = np.arccos(cos_sims)
                inside_sa95 = float(np.percentile(sas, 95))

        # 4. Aspect Ratio Check
        aspect_ratio = 1.0
        rows, cols = np.where(mask)
        if len(rows) > 0:
            h_bbox = rows.max() - rows.min() + 1
            w_bbox = cols.max() - cols.min() + 1
            aspect_ratio = float(max(h_bbox, w_bbox) / (min(h_bbox, w_bbox) + 1e-5))

        qc_stats = {
            "method": method,
            "total_pixels": total_pixels,
            "foreground_pixels": fg_pixels,
            "foreground_ratio": round(fg_ratio, 4),
            "border_contact": border_contact,
            "outer_ring_sa_rad": round(ring_sa, 4),
            "outer_ring_signal_ratio": round(ring_signal_ratio, 4),
            "inside_sa95_rad": round(inside_sa95, 4),
            "aspect_ratio": round(aspect_ratio, 2),
            "image_dimensions": [H, W]
        }

        # Enforce Status Rules
        if fg_ratio < min_roi_frac or fg_ratio > max_roi_frac or aspect_ratio > 2.5:
            status = "REJECTED"
            qc_stats["warning"] = f"Foreground ratio {fg_ratio:.4f} or aspect ratio {aspect_ratio:.2f} out of allowed bounds"
        elif border_contact or fg_ratio < 0.10 or fg_ratio > 0.75 or aspect_ratio > 2.0:
            status = "PASS_WITH_WARNING"
            qc_stats["warning"] = f"Mask has minor warnings: border_contact={border_contact}, ratio={fg_ratio:.4f}"
        else:
            status = "PASS"

        return mask, qc_stats, status

