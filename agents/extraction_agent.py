from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.scientific_tools.extraction import ExtractionTool

class ExtractionAgent:
    """
    Spectral Extraction Agent:
    Accepts ROI mask and HSI cube, filters invalid/saturated pixels, extracts clean spectrum,
    and reports spectral signal quality.
    """
    
    SYSTEM_PROMPT = """
    You are the Spectral Extraction Agent. Your role is to inspect ROI spectra extracted from hyperspectral cubes.
    Ensure invalid/saturated pixels are removed and SNR is sufficient.
    
    Respond in JSON format:
    {
      "action": "ACCEPT_SPECTRUM" | "REQUEST_RESEGMENTATION" | "REJECT_SAMPLE",
      "rationale": "EXPLANATION",
      "status": "PASS" | "PASS_WITH_WARNING" | "REJECTED"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def process_extraction(self, cube, mask, sample_id="unknown"):
        spectrum, qc, status = ExtractionTool.extract_spectrum(cube, mask, sample_id=sample_id)
        
        user_prompt = f"Sample ID: {sample_id}\nExtraction Status: '{status}'\nQC Diagnostics: {qc}"
        llm_res = self.client.query_json(self.SYSTEM_PROMPT, user_prompt)
        
        decision = llm_res["parsed"] if llm_res["success"] else {
            "action": "ACCEPT_SPECTRUM" if status != "REJECTED" else "REJECT_SAMPLE",
            "rationale": f"Extraction completed with status {status}",
            "status": status
        }
        
        return spectrum, qc, decision["status"], decision, llm_res.get("latency_sec", 0.0)
