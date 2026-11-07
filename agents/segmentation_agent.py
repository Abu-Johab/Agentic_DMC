from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.scientific_tools.segmentation import SegmentationTool

class SegmentationAgent:
    """
    Segmentation Specialist Agent:
    Inspects image diagnostics, selects approved segmentation algorithms
    (noise_floor, noise_floor_hull, multi_otsu, grabcut, otsu, connected_components),
    evaluates quality metrics, and decides whether to accept the ROI mask or retry an alternative method.
    """
    
    SYSTEM_PROMPT = """
    You are the Segmentation Agent. Your goal is to segment mushroom HSI cubes to generate an accurate foreground mask.
    Allowed segmentation methods: ["noise_floor", "noise_floor_hull", "multi_otsu", "grabcut", "otsu", "connected_components", "band_ratio", "pca_threshold"].
    Allowed foreground ratio range: [0.05, 0.85].
    
    Analyze the provided diagnostic statistics and choose the best action.
    Respond in JSON format:
    {
      "selected_method": "noise_floor" | "noise_floor_hull" | "multi_otsu" | "grabcut" | "otsu" | "connected_components",
      "action": "ACCEPT" | "RETRY_ALTERNATIVE" | "REJECT",
      "rationale": "EXPLANATION",
      "status": "PASS" | "PASS_WITH_WARNING" | "REJECTED"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def process_cube(self, cube, sample_id="unknown", max_retries=3):
        attempts = []
        allowed_methods = ["noise_floor", "noise_floor_hull", "multi_otsu", "connected_components", "otsu"]
        current_method = "noise_floor"
        total_latency = 0.0
        
        for idx in range(max_retries):
            mask, qc, det_status = SegmentationTool.segment_cube(cube, method=current_method)
            
            user_prompt = (
                f"Sample ID: {sample_id}\nAttempt {idx+1}/{max_retries}\n"
                f"Executed Method: '{current_method}'\nDeterministic Status: '{det_status}'\n"
                f"QC Diagnostics: {qc}\nSelect action and next method if retrying."
            )
            llm_res = self.client.query_json(self.SYSTEM_PROMPT, user_prompt)
            lat = llm_res.get("latency_sec", 0.001)
            total_latency += lat
            
            if llm_res["success"] and isinstance(llm_res.get("parsed"), dict):
                parsed = llm_res["parsed"]
                action = parsed.get("action", "ACCEPT")
                proposed_method = parsed.get("selected_method", current_method)
                llm_status = parsed.get("status", det_status)
                rationale = parsed.get("rationale", f"Evaluated attempt {idx+1}")
                is_fallback = False
            else:
                action = "ACCEPT" if det_status in ["PASS", "PASS_WITH_WARNING"] else "RETRY_ALTERNATIVE"
                proposed_method = allowed_methods[(allowed_methods.index(current_method) + 1) % len(allowed_methods)]
                llm_status = det_status
                rationale = f"Deterministic fallback rule evaluated for {current_method}"
                is_fallback = True
                
            # Deterministic QC gate is authoritative over LLM status
            final_status = det_status if det_status == "REJECTED" else llm_status
            if det_status == "REJECTED" and final_status == "PASS":
                final_status = "REJECTED"
                
            attempt_record = {
                "attempt": idx + 1,
                "executed_method": current_method,
                "proposed_method": proposed_method,
                "action": action,
                "det_status": det_status,
                "final_status": final_status,
                "qc": qc,
                "rationale": rationale,
                "is_fallback": is_fallback
            }
            attempts.append(attempt_record)
            
            if action == "ACCEPT" and final_status in ["PASS", "PASS_WITH_WARNING"]:
                return mask, qc, final_status, attempts, round(total_latency, 3)
                
            if action == "RETRY_ALTERNATIVE" and idx < max_retries - 1:
                if proposed_method in allowed_methods and proposed_method != current_method:
                    current_method = proposed_method
                else:
                    current_method = allowed_methods[(allowed_methods.index(current_method) + 1) % len(allowed_methods)]
                continue
                
            if action == "REJECT" or final_status == "REJECTED":
                return mask, qc, "REJECTED", attempts, round(total_latency, 3)
                
        # Return last attempt result if max_retries reached
        last_attempt = attempts[-1]
        return mask, last_attempt["qc"], last_attempt["final_status"], attempts, round(total_latency, 3)


