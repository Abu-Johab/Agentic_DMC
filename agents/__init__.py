"""
Standalone Agent Definitions for Autonomous Multi-Agent DMC System
"""
from .supervisor_agent import SupervisorAgent
from .preprocessing_agent import PreprocessingAgent
from .modeling_agent import ModelingAgent
from .statistical_agent import StatisticalAgent
from .explainability_agent import ExplainabilityAgent
from .spatial_agent import SpatialAgent
from .stability_agent import StabilityAgent
from .segmentation_agent import SegmentationAgent
from .extraction_agent import ExtractionAgent

__all__ = [
    "SupervisorAgent",
    "PreprocessingAgent",
    "ModelingAgent",
    "StatisticalAgent",
    "ExplainabilityAgent",
    "SpatialAgent",
    "StabilityAgent",
    "SegmentationAgent",
    "ExtractionAgent"
]
