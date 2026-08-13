"""
Vision Pipeline for Crowd Flow Optimiser.
Handles YOLO, Depth, and VLM model integration for real-time video analytics.
All models are managed through the centralized ModelManager.
"""
from .model_manager import model_manager
