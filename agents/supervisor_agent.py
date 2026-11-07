from agentic_dmc.orchestration.llm_client import OllamaClient

class SupervisorAgent:
    """
    Supervisor / Planner Agent:
    Oversees multi-agent pipeline execution, verifies prerequisites, evaluates stage status,
    and routes actions across specialist agents.
    """
    
    SYSTEM_PROMPT = """
    You are the Supervisor Agent in an agentic multi-agent hyperspectral Dry Matter Content (DMC) prediction system.
    Your objective is to direct the specialist agents through a scientifically rigorous, leakage-free workflow:
    1. INITIALIZE_AUDIT: Audit data and checksums
    2. SEGMENTATION_AND_EXTRACTION: Process HSI image cubes into clean ROI masks and spectra
    3. PREPROCESSING_AND_MODEL_TRAINING: Optimally search preprocessing and ML regression models on calibration split
    4. PIPELINE_FREEZING: Freeze approved preprocessor and model package
    5. SPATIAL_MAPPING: Run pixel-wise DMC spatial estimation on HSI cubes
    6. SYSTEM_EVALUATION: Evaluate regression metrics, agent reliability, and safety guardrails
    
    Respond in JSON format with fields:
    {
      "next_stage": "STAGE_NAME",
      "action": "ACTION_DESCRIPTION",
      "rationale": "REASONING",
      "status": "PASS" | "PASS_WITH_WARNING" | "WITHHELD" | "REJECTED"
    }
    """
    
    ALLOWED_STAGES = {
        "INITIALIZE_AUDIT",
        "SEGMENTATION_AND_EXTRACTION",
        "PREPROCESSING_AND_MODEL_TRAINING",
        "PIPELINE_FREEZING",
        "SPATIAL_MAPPING",
        "EXPLAINABILITY",
        "SYSTEM_EVALUATION",
        "COMPLETE"
    }
    ALLOWED_STATUSES = {"PASS", "PASS_WITH_WARNING", "WITHHELD", "REJECTED"}

    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def plan_next_step(self, current_stage, stage_outcomes):
        prompt = f"Current Stage: {current_stage}\nCompleted Outcomes: {stage_outcomes}\nSelect the next valid stage or action."
        res = self.client.query_json(self.SYSTEM_PROMPT, prompt)
        
        fallback_map = {
            "INIT": "SEGMENTATION_AND_EXTRACTION",
            "INITIALIZE_AUDIT": "SEGMENTATION_AND_EXTRACTION",
            "SEGMENTATION_AND_EXTRACTION": "PREPROCESSING_AND_MODEL_TRAINING",
            "PREPROCESSING_AND_MODEL_TRAINING": "PIPELINE_FREEZING",
            "PIPELINE_FREEZING": "SPATIAL_MAPPING",
            "SPATIAL_MAPPING": "EXPLAINABILITY",
            "EXPLAINABILITY": "SYSTEM_EVALUATION",
            "SYSTEM_EVALUATION": "COMPLETE"
        }

        if res["success"] and isinstance(res.get("parsed"), dict):
            parsed = res["parsed"]
            next_stage = parsed.get("next_stage")
            status = parsed.get("status")
            action = parsed.get("action", f"Proceed to {next_stage}")
            rationale = parsed.get("rationale", "Validated LLM plan")
            
            if next_stage in self.ALLOWED_STAGES and status in self.ALLOWED_STATUSES:
                return {
                    "next_stage": next_stage,
                    "action": action,
                    "rationale": rationale,
                    "status": status,
                    "decision_source": "LLM"
                }, res["latency_sec"]

        # Deterministic fallback when LLM output is unavailable or malformed/invalid
        if current_stage not in fallback_map and current_stage not in self.ALLOWED_STAGES:
            next_st = "FAILED_STAGE"
            status = "REJECTED"
            rationale = f"Unknown current_stage '{current_stage}' rejected by supervisor safety gate."
        else:
            next_st = fallback_map.get(current_stage, "COMPLETE")
            status = "PASS" if next_st != "FAILED_STAGE" else "REJECTED"
            rationale = f"Approved deterministic fallback rule executed for stage {current_stage}"
            
        return {
            "next_stage": next_st,
            "action": f"Proceed to {next_st}",
            "rationale": rationale,
            "status": status,
            "decision_source": "DETERMINISTIC_FALLBACK"
        }, res.get("latency_sec", 0.001)

