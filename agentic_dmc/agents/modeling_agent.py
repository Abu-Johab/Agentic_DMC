from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.scientific_tools.modeling import ModelingTool

class ModelingAgent:
    """
    Modeling & Prediction Specialist Agent:
    Configures ML model hyperparameter searches (PLSR, SVR, RandomForest, CNN1D),
    evaluates cross-validation metrics, enforces early stopping, and approves final model package.
    """
    
    SYSTEM_PROMPT = """
    You are the Modeling/Prediction Agent. Your goal is to evaluate machine learning regression algorithms for DMC prediction.
    Allowed algorithms: ["PLSR", "SVR", "RandomForest", "CNN1D"].
    
    Inspect the model comparison table and select the best model package based on CV RMSECV and Validation RMSEV.
    Respond in JSON format:
    {
      "selected_model": "PLSR" | "SVR" | "RandomForest" | "CNN1D",
      "action": "FREEZE_MODEL",
      "rationale": "SCIENTIFIC JUSTIFICATION",
      "status": "PASS"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def select_best_model(self, model_eval_summary):
        prompt = f"Model Evaluation Summary on Calibration & Validation splits:\n{model_eval_summary}\nSelect the top-performing model package."
        llm_res = self.client.query_json(self.SYSTEM_PROMPT, prompt)
        
        if llm_res["success"] and "selected_model" in llm_res["parsed"]:
            best_model_name = llm_res["parsed"]["selected_model"]
            decision = llm_res["parsed"]
        else:
            # Fallback: pick model with lowest validation RMSE
            best_model_name = min(model_eval_summary.items(), key=lambda x: x[1].get("rmsev", float("inf")))[0]
            decision = {
                "selected_model": best_model_name,
                "action": "FREEZE_MODEL",
                "rationale": f"Selected {best_model_name} with lowest RMSEV {model_eval_summary[best_model_name]['rmsev']:.4f}",
                "status": "PASS"
            }
            
        return best_model_name, decision, llm_res.get("latency_sec", 0.0)
