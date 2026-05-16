#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Model Configuration Examples

This file shows how to create configurations for different Ollama models.
Supported popular models include: Llama2, Mistral, CodeLlama, Qwen, etc.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_beer_game.config import GameConfig, LLMConfig, create_standard_config

# Popular Ollama model configurations
OLLAMA_MODELS = {
    "llama2": {
        "model": "llama2",
        "description": "Meta's Llama2 model, balanced performance and quality",
        "temperature": 0.3,
        "max_tokens": 200,
        "timeout": 60
    },
    "llama2:13b": {
        "model": "llama2:13b",
        "description": "Llama2 13B version, higher quality but requires more resources",
        "temperature": 0.2,
        "max_tokens": 250,
        "timeout": 90
    },
    "mistral": {
        "model": "mistral",
        "description": "Mistral AI's efficient model",
        "temperature": 0.4,
        "max_tokens": 180,
        "timeout": 45
    },
    "codellama": {
        "model": "codellama",
        "description": "Llama model specialized for code generation",
        "temperature": 0.1,
        "max_tokens": 300,
        "timeout": 75
    },
    "qwen": {
        "model": "qwen:7b",
        "description": "Alibaba's Tongyi Qianwen model",
        "temperature": 0.3,
        "max_tokens": 200,
        "timeout": 60
    },
    "gemma": {
        "model": "gemma:7b",
        "description": "Google's Gemma model",
        "temperature": 0.3,
        "max_tokens": 200,
        "timeout": 60
    },
    "phi": {
        "model": "phi",
        "description": "Microsoft's Phi model, small and efficient",
        "temperature": 0.4,
        "max_tokens": 150,
        "timeout": 30
    },
    "neural-chat": {
        "model": "neural-chat",
        "description": "Intel-optimized chat model",
        "temperature": 0.5,
        "max_tokens": 200,
        "timeout": 60
    }
}

def create_ollama_config_for_model(model_key: str, base_url: str = "http://192.168.31.209:11434") -> GameConfig:
    """
    Create game config for specified Ollama model

    Args:
        model_key: Model key name (defined in OLLAMA_MODELS)
        base_url: Ollama service address

    Returns:
        Game config object

    Raises:
        ValueError: If model key does not exist
    """
    if model_key not in OLLAMA_MODELS:
        available_models = ", ".join(OLLAMA_MODELS.keys())
        raise ValueError(f"Unknown model: {model_key}. Available models: {available_models}")

    model_config = OLLAMA_MODELS[model_key]

    # Create base config
    config = create_standard_config()

    # Configure Ollama LLM
    ollama_config = LLMConfig(
        provider="ollama",
        model=model_config["model"],
        base_url=base_url,
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        timeout=model_config["timeout"]
    )

    # Set same LLM config for all agents
    config.retailer.llm_config = ollama_config
    config.wholesaler.llm_config = ollama_config
    config.distributor.llm_config = ollama_config
    config.manufacturer.llm_config = ollama_config

    return config

def create_mixed_ollama_config(base_url: str = "http://192.168.31.209:11434") -> GameConfig:
    """
    Create mixed config using different Ollama models
    Different roles use different models to demonstrate diversity

    Args:
        base_url: Ollama service address

    Returns:
        Game config object
    """
    config = create_standard_config()

    # Retailer uses fast-response model
    config.retailer.llm_config = LLMConfig(
        provider="ollama",
        model="phi",
        base_url=base_url,
        temperature=0.4,
        max_tokens=150,
        timeout=30
    )

    # Wholesaler uses balanced model
    config.wholesaler.llm_config = LLMConfig(
        provider="ollama",
        model="mistral",
        base_url=base_url,
        temperature=0.3,
        max_tokens=180,
        timeout=45
    )

    # Distributor uses general-purpose model
    config.distributor.llm_config = LLMConfig(
        provider="ollama",
        model="llama2",
        base_url=base_url,
        temperature=0.3,
        max_tokens=200,
        timeout=60
    )

    # Manufacturer uses high-quality model (requires more complex decisions)
    config.manufacturer.llm_config = LLMConfig(
        provider="ollama",
        model="qwen:7b",
        base_url=base_url,
        temperature=0.2,
        max_tokens=250,
        timeout=75
    )

    return config

def create_fallback_config(primary_model: str = "llama2",
                          fallback_model: str = "phi",
                          base_url: str = "http://localhost:11434") -> GameConfig:
    """
    Create config with fallback model
    Automatically switches to fallback if primary model is unavailable

    Args:
        primary_model: Primary model
        fallback_model: Fallback model
        base_url: Ollama service address

    Returns:
        Game config object
    """
    config = create_standard_config()

    # Primary configuration
    primary_config = OLLAMA_MODELS.get(primary_model, OLLAMA_MODELS["llama2"])
    fallback_config_data = OLLAMA_MODELS.get(fallback_model, OLLAMA_MODELS["phi"])

    # Create LLM config with fallback
    llm_config = LLMConfig(
        provider="ollama",
        model=primary_config["model"],
        base_url=base_url,
        temperature=primary_config["temperature"],
        max_tokens=primary_config["max_tokens"],
        timeout=primary_config["timeout"],
        fallback_configs=[
            {
                "provider": "ollama",
                "model": fallback_config_data["model"],
                "base_url": base_url,
                "temperature": fallback_config_data["temperature"],
                "max_tokens": fallback_config_data["max_tokens"],
                "timeout": fallback_config_data["timeout"]
            }
        ]
    )

    # Set same config for all agents
    config.retailer.llm_config = llm_config
    config.wholesaler.llm_config = llm_config
    config.distributor.llm_config = llm_config
    config.manufacturer.llm_config = llm_config

    return config

def print_available_models():
    """
    Print all available Ollama model configurations
    """
    print("🤖 Available Ollama Model Configurations:")
    print("=" * 60)

    for key, model_info in OLLAMA_MODELS.items():
        print(f"\n📦 {key}")
        print(f"   Model: {model_info['model']}")
        print(f"   Description: {model_info['description']}")
        print(f"   Temperature: {model_info['temperature']}")
        print(f"   Max tokens: {model_info['max_tokens']}")
        print(f"   Timeout: {model_info['timeout']}s")

    print("\n💡 Usage:")
    print("   config = create_ollama_config_for_model('llama2')")
    print("   config = create_mixed_ollama_config()")
    print("   config = create_fallback_config('mistral', 'phi')")

def get_model_download_commands():
    """
    Get model download commands

    Returns:
        Dictionary of model download commands
    """
    commands = {}
    for key, model_info in OLLAMA_MODELS.items():
        model_name = model_info["model"]
        commands[key] = f"ollama pull {model_name}"

    return commands

def print_download_commands():
    """
    Print download commands for all models
    """
    print("📥 Model Download Commands:")
    print("=" * 40)

    commands = get_model_download_commands()
    for key, command in commands.items():
        model_info = OLLAMA_MODELS[key]
        print(f"\n# {model_info['description']}")
        print(f"{command}")

    print("\n⚠️  Note: Large models may require several GB of space and take considerable time to download")

if __name__ == "__main__":
    print_available_models()
    print("\n" + "=" * 60)
    print_download_commands()