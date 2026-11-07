import json
import time
from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.explainability.explainability_tool import ExplainabilityTool

class ExplainabilityAgent:
    """
    Explainability Specialist Agent:
    Autonomous decision agent responsible for selecting explanation methods,
    executing SHAP computations, inspecting prediction consistency and additivity residuals,
    evaluating rank stability, and approving explainability reports and figures.
    """
    
    SYSTEM_PROMPT = """
    You are the Explainability Selection & Verification Agent for Hyperspectral DMC Prediction.
    Your role is to evaluate numerical SHAP attributions, check prediction consistency and additivity diagnostics,
    and verify spectral feature rankings.
    
    Allowed Explanation Modes: ["Mode_A_Original_Spectrum", "Mode_B_Model_Input"].
    Allowed Explainer Types: ["KernelExplainer", "DeepExplainer", "PermutationExplainer"].
    
    Evaluate the provided diagnostic validation statistics and choose the best action.
    Respond strictly in JSON format:
    {
      "selected_mode": "Mode_A_Original_Spectrum",
      "action": "APPROVE_EXPLANATION" | "RETRY_WITH_ADJUSTED_SETTINGS" | "REJECT",
      "rationale": "REASONING BASED ON PREDICTION CONSISTENCY, ADDITIVITY RESIDUALS, AND STABILITY",
      "status": "PASS" | "PASS_WITH_WARNING" | "REJECTED"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def process_explainability(self, n_clusters=25, nsamples=300, random_seed=42):
        """
        Executes observation-action-verification loop for SHAP explainability.
        """
        t0 = time.time()
        attempts = []
        
        # 1. Execute numerical SHAP computation tool
        validation_diag = ExplainabilityTool.run_full_explainability_pipeline(
            n_clusters=n_clusters, nsamples=nsamples, random_seed=random_seed
        )
        latency = round(time.time() - t0, 3)
        
        # 2. Query LLM agent to confirm decision
        user_prompt = f"Validation Diagnostics:\n{json.dumps(validation_diag, indent=2)}\nExecution Time: {latency}s"
        llm_res = self.client.query_json(self.SYSTEM_PROMPT, user_prompt)
        
        if llm_res["success"] and "action" in llm_res["parsed"]:
            decision = llm_res["parsed"]
        else:
            overall_pass = validation_diag.get("overall_validation") == "PASSED"
            decision = {
                "selected_mode": "Mode_A_Original_Spectrum",
                "action": "APPROVE_EXPLANATION" if overall_pass else "REJECT",
                "rationale": f"Numerical checks passed: consistency={validation_diag['prediction_consistency']['passed']}, additivity={validation_diag['reconstruction_additivity']['passed']}",
                "status": "PASS" if overall_pass else "REJECTED"
            }
            
        attempts.append({
            "mode": "Mode_A_Original_Spectrum",
            "diagnostics": validation_diag,
            "decision": decision
        })
        
        return validation_diag, decision, latency, attempts
