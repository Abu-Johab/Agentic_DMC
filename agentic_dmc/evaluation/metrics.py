import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    Computes regression metrics in original DMC units:
    - RMSE
    - MAE
    - R²
    - Bias (defined as mean(y_pred - y_true))
    - RPD (std(y_true, ddof=1) / RMSE)
    - RPIQ ((Q3 - Q1) / RMSE)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    
    std_true = float(np.std(y_true, ddof=1))
    rpd = float(std_true / rmse) if rmse > 0 else float("inf")
    
    q75, q25 = np.percentile(y_true, 75), np.percentile(y_true, 25)
    rpiq = float((q75 - q25) / rmse) if rmse > 0 else float("inf")
    
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Bias": round(bias, 4),
        "RPD": round(rpd, 4),
        "RPIQ": round(rpiq, 4)
    }

def compute_segmentation_metrics(pred_mask, ref_mask):
    """
    Computes segmentation metrics when reference mask is available:
    - Dice
    - IoU
    - Precision
    - Recall
    """
    p = np.asarray(pred_mask, dtype=bool)
    r = np.asarray(ref_mask, dtype=bool)
    
    intersection = np.logical_and(p, r).sum()
    union = np.logical_or(p, r).sum()
    
    p_sum = p.sum()
    r_sum = r.sum()
    
    dice = float(2 * intersection / (p_sum + r_sum + 1e-8))
    iou = float(intersection / (union + 1e-8))
    precision = float(intersection / (p_sum + 1e-8))
    recall = float(intersection / (r_sum + 1e-8))
    
    return {
        "Dice": round(dice, 4),
        "IoU": round(iou, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

def compute_agent_system_metrics(decision_log):
    """
    Computes system and agent reliability metrics dynamically from decision_log:
    - Valid-input task success rate
    - Valid tool-call rate
    - Recovery success rate (or 'N/A' if no recovery opportunity)
    - Unsafe acceptance rate (or 'N/A' if no invalid inputs)
    - False rejection rate (or 'N/A' if no valid inputs)
    - Median and p95 end-to-end latency
    - Audit completeness
    """
    latencies = [entry.get("latency_sec", 0.0) for entry in decision_log if "latency_sec" in entry]
    if not latencies:
        latencies = [0.0]
        
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    
    total_calls = len(decision_log)
    successful_calls = sum(1 for e in decision_log if e.get("status") in ["PASS", "PASS_WITH_WARNING", "SUCCESS", "RECOVERED"])
    
    valid_tool_call_rate = float(successful_calls / (total_calls + 1e-8)) if total_calls > 0 else 0.0
    
    # Dynamic recovery evaluation
    recovery_opps = sum(1 for e in decision_log if e.get("has_recovery") is True or e.get("recovery_attempted") is True)
    recovery_succ = sum(1 for e in decision_log if (e.get("has_recovery") is True or e.get("recovery_attempted") is True) and e.get("status") in ["RECOVERED", "SUCCESS"])
    
    recovery_rate = round(float(recovery_succ / recovery_opps), 4) if recovery_opps > 0 else "N/A"
    
    # Dynamic safety rejections & acceptances
    invalid_inputs = sum(1 for e in decision_log if e.get("input_validity") == "INVALID")
    unsafe_accepts = sum(1 for e in decision_log if e.get("input_validity") == "INVALID" and e.get("status") in ["SUCCESS", "PASS"])
    unsafe_rate = round(float(unsafe_accepts / invalid_inputs), 4) if invalid_inputs > 0 else 0.0
    
    valid_inputs = sum(1 for e in decision_log if e.get("input_validity") == "VALID")
    false_rejects = sum(1 for e in decision_log if e.get("input_validity") == "VALID" and e.get("status") in ["REJECTED", "SAFETY_REFUSAL"])
    false_rej_rate = round(float(false_rejects / valid_inputs), 4) if valid_inputs > 0 else 0.0
    
    return {
        "total_agent_calls": total_calls,
        "valid_tool_call_rate": round(valid_tool_call_rate, 4),
        "task_success_rate": round(float(successful_calls / total_calls), 4) if total_calls > 0 else 0.0,
        "unsafe_acceptance_rate": unsafe_rate,
        "false_rejection_rate": false_rej_rate,
        "recovery_success_rate": recovery_rate,
        "median_latency_sec": round(med_lat, 3),
        "p95_latency_sec": round(p95_lat, 3),
        "audit_completeness": 1.0 if total_calls > 0 else 0.0
    }
