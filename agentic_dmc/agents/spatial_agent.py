from agentic_dmc.orchestration.llm_client import OllamaClient
from agentic_dmc.scientific_tools.spatial import SpatialTool

class SpatialAgent:
    """
    Spatial Visualization Agent:
    Validates cube dimensions, executes pixel-wise DMC predictions, checks out-of-domain layers,
    and approves spatial mapping artifacts.
    """
    
    SYSTEM_PROMPT = """
    You are the Spatial Visualization Agent. Your role is to inspect pixel-wise DMC spatial maps.
    Ensure out-of-domain background pixels are masked and spatial DMC range is realistic (0.0 to 1.0).
    Label output as 'model-estimated spatial DMC distribution'.
    
    Respond in JSON format:
    {
      "action": "APPROVE_SPATIAL_MAP" | "REJECT_SPATIAL_MAP",
      "label": "model-estimated spatial DMC distribution",
      "rationale": "EXPLANATION",
      "status": "PASS" | "PASS_WITH_WARNING" | "REJECTED"
    }
    """
    
    def __init__(self, llm_client=None):
        self.client = llm_client or OllamaClient()

    def process_spatial(self, cube, mask, pipeline_pkg, sample_id="M160", output_dir="agentic_dmc/outputs/spatial_maps", true_dmc=None):
        dmc_map, qc, status = SpatialTool.generate_spatial_maps(cube, mask, pipeline_pkg, sample_id=sample_id, output_dir=output_dir, true_dmc=true_dmc)
        
        prompt = f"Sample ID: {sample_id}\nSpatial QC Stats: {qc}\nStatus: '{status}'"
        llm_res = self.client.query_json(self.SYSTEM_PROMPT, prompt)
        
        decision = llm_res["parsed"] if llm_res["success"] else {
            "action": "APPROVE_SPATIAL_MAP" if status != "REJECTED" else "REJECT_SPATIAL_MAP",
            "label": "model-estimated spatial DMC distribution",
            "rationale": f"Spatial map generation completed with status {status}",
            "status": status
        }
        
        return dmc_map, qc, decision["status"], decision, llm_res.get("latency_sec", 0.0)

