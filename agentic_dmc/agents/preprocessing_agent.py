from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.scientific_tools.preprocessing import PreprocessingTool

class PreprocessingAgent:
    """
    Preprocessing Specialist Agent:
    Evaluates bounded preprocessing pipeline candidates ("raw", "snv", "msc", "sg_smooth", "sg_deriv1", "snv_sg1", "msc_sg1"),
    interprets cross-validation metrics, and selects the optimal preprocessing transformation balancing low CV RMSE
    and cross-modal HSI transferability.
    """
    
    SYSTEM_PROMPT = """
    You are the Preprocessing Selection Agent. Your goal is to select the optimal spectral preprocessing pipeline for mushroom HSI DMC prediction.
    Allowed candidates: ["raw", "snv", "msc", "sg_smooth", "sg_deriv1", "snv_sg1", "msc_sg1"].
    
    Guidelines:
    1. Evaluate cross-validation RMSECV and RMSEV metrics.
    2. Prefer scatter-corrected and baseline-corrected methods (snv_sg1, snv, msc_sg1) when their RMSECV is within 5% of the minimum candidate RMSECV. Scatter and baseline correction is scientifically required to eliminate illumination variations between single CSV spectra and pixel-level 3D HSI datacubes.
    
    Respond strictly in JSON format:
    {
      "selected_preprocessing": "METHOD_NAME",
      "action": "SELECT_PREPROCESSING",
      "rationale": "REASONING BASED ON CV RMSE AND CROSS-MODAL HSI TRANSFERABILITY",
      "status": "PASS"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def select_best_preprocessing(self, cv_results_dict):
        prompt = f"Candidate Preprocessing CV Results:\n{cv_results_dict}\nSelect the optimal preprocessing method balancing low RMSECV and HSI deployment transferability."
        llm_res = self.client.query_json(self.SYSTEM_PROMPT, prompt)
        
        if llm_res["success"] and "selected_preprocessing" in llm_res["parsed"]:
            best_prep = llm_res["parsed"]["selected_preprocessing"]
            decision = llm_res["parsed"]
        else:
            # Fallback: prefer snv_sg1 if its RMSECV is within 5% of the absolute minimum RMSECV, else pick min RMSECV
            min_method, min_res = min(cv_results_dict.items(), key=lambda x: x[1].get("rmsecv", float("inf")))
            min_rmsecv = min_res.get("rmsecv", float("inf"))
            
            if "snv_sg1" in cv_results_dict and (cv_results_dict["snv_sg1"]["rmsecv"] <= min_rmsecv * 1.05):
                best_prep = "snv_sg1"
                rationale = f"Selected snv_sg1 (RMSECV={cv_results_dict['snv_sg1']['rmsecv']:.4f}) for optimal HSI cross-modal scatter correction and transferability."
            else:
                best_prep = min_method
                rationale = f"Selected {best_prep} with lowest RMSECV {min_rmsecv:.4f}"
                
            decision = {
                "selected_preprocessing": best_prep,
                "action": "SELECT_PREPROCESSING",
                "rationale": rationale,
                "status": "PASS"
            }
            
        return best_prep, decision, llm_res.get("latency_sec", 0.0)

