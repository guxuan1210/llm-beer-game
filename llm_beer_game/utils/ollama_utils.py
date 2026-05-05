#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Utility Functions

Provides functions for getting available Ollama model lists, etc.
"""

import requests
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_available_ollama_models(base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """
    Get all available model list from Ollama

    Args:
        base_url: Ollama service address

    Returns:
        Model list, each model contains name, size, etc.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            # Sort by model name
            models.sort(key=lambda x: x.get("name", ""))

            return models
        else:
            logger.warning(f"Failed to get Ollama model list, status code: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to connect to Ollama service: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting model list: {e}")
        return []

def get_model_names_only(base_url: str = "http://localhost:11434") -> List[str]:
    """
    Get list of all available model names from Ollama

    Args:
        base_url: Ollama service address

    Returns:
        List of model names
    """
    models = get_available_ollama_models(base_url)
    return [model.get("name", "") for model in models if model.get("name")]

def check_ollama_service(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama service is available

    Args:
        base_url: Ollama service address

    Returns:
        Whether the service is available
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def format_model_size(size_bytes: int) -> str:
    """
    Format model size for display

    Args:
        size_bytes: Model size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "Unknown size"

    # Convert to GB
    size_gb = size_bytes / (1024 ** 3)
    if size_gb >= 1:
        return f"{size_gb:.1f} GB"

    # Convert to MB
    size_mb = size_bytes / (1024 ** 2)
    return f"{size_mb:.1f} MB"

def get_model_display_info(base_url: str = "http://localhost:11434") -> List[Dict[str, str]]:
    """
    Get model info for UI display

    Args:
        base_url: Ollama service address

    Returns:
        List of models with display info
    """
    models = get_available_ollama_models(base_url)
    display_info = []

    for model in models:
        name = model.get("name", "")
        size = model.get("size", 0)

        if name:
            display_info.append({
                "name": name,
                "display_name": f"{name} ({format_model_size(size)})",
                "size": size,
                "size_formatted": format_model_size(size)
            })

    return display_info

# Predefined recommended models (used when dynamic list cannot be obtained)
RECOMMENDED_MODELS = [
    "gemma3:27b",  # Priority recommendation
    "llama2",
    "llama2:7b",
    "llama2:13b",
    "mistral",
    "mistral:7b",
    "codellama",
    "codellama:7b",
    "qwen",
    "qwen:7b",
    "qwen:14b",
    "gemma",
    "gemma:7b",
    "phi",
    "phi:3b",
    "neural-chat",
    "gpt-oss:20b"
]

def get_models_with_fallback(base_url: str = "http://localhost:11434") -> List[str]:
    """
    Get model list, fall back to recommended models if connection fails

    Args:
        base_url: Ollama service address

    Returns:
        List of model names
    """
    # Try to get actually available models first
    available_models = get_model_names_only(base_url)

    if available_models:
        return available_models
    else:
        # Fall back to recommended model list
        logger.info("Cannot get Ollama model list, using recommended model list")
        return RECOMMENDED_MODELS