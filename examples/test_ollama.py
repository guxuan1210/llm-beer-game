#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Connection Test Script

Tests whether the Ollama service is running properly and models are available.
Run this script to quickly verify Ollama configuration.
"""

import sys
import requests
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_beer_game.llm import LLMClientFactory

def check_ollama_service(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama service is running

    Args:
        base_url: Ollama service address

    Returns:
        Whether service is available
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Cannot connect to Ollama service: {e}")
        return False

def get_available_models(base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """
    Get list of installed Ollama models

    Args:
        base_url: Ollama service address

    Returns:
        List of models
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        else:
            return []
    except Exception as e:
        print(f"❌ Failed to get model list: {e}")
        return []

def test_model_generation(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """
    Test model generation capability

    Args:
        model_name: Model name
        base_url: Ollama service address

    Returns:
        Whether test succeeded
    """
    try:
        client = LLMClientFactory.create_client(
            "ollama",
            model_name=model_name,
            base_url=base_url
        )

        if not client.is_available():
            print(f"❌ Model {model_name} unavailable")
            return False

        # Simple test prompt
        response = client.generate(
            system_prompt="You are a supply chain management assistant.",
            user_prompt="What is inventory management? Answer in one sentence.",
            temperature=0.1,
            max_tokens=50
        )

        print(f"✅ Model {model_name} test succeeded")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return True

    except Exception as e:
        print(f"❌ Model {model_name} test failed: {e}")
        return False

def print_installation_guide():
    """
    Print Ollama installation guide
    """
    print("\n📖 Ollama Installation Guide:")
    print("=" * 40)
    print("\n1. Download and install Ollama:")
    print("   https://ollama.ai/")
    print("\n2. Start Ollama service:")
    print("   ollama serve")
    print("\n3. Download models (choose one or more):")
    print("   ollama pull llama2        # General purpose")
    print("   ollama pull mistral       # Efficient model")
    print("   ollama pull qwen:7b       # Chinese-friendly")
    print("   ollama pull phi           # Lightweight model")
    print("   ollama pull codellama     # Code specialized")
    print("\n4. Verify installation:")
    print("   python examples/test_ollama.py")

def print_model_recommendations():
    """
    Print model recommendations
    """
    print("\n🎯 Model Recommendations:")
    print("=" * 30)
    print("\n🚀 Quick Start (recommended for beginners):")
    print("   ollama pull phi           # Small, fast, good for testing")
    print("\n⚖️  Balanced Choice:")
    print("   ollama pull llama2        # Balanced performance and quality")
    print("   ollama pull mistral       # Efficient with good quality")
    print("\n🎯 Chinese Optimized:")
    print("   ollama pull qwen:7b       # Alibaba Tongyi Qianwen, good Chinese performance")
    print("\n💻 Code Tasks:")
    print("   ollama pull codellama     # Specialized for code generation")
    print("\n⚡ High Performance (requires more resources):")
    print("   ollama pull llama2:13b    # Larger model, better results")

def main():
    """
    Main test function
    """
    print("🔍 Ollama Connection Test")
    print("=" * 30)

    base_url = "http://localhost:11434"

    # 1. Check service status
    print("\n1️⃣ Checking Ollama service...")
    if not check_ollama_service(base_url):
        print("\n❌ Ollama service not running or unreachable")
        print_installation_guide()
        return

    print("✅ Ollama service running normally")

    # 2. Get installed models
    print("\n2️⃣ Getting installed models...")
    models = get_available_models(base_url)

    if not models:
        print("❌ No installed models found")
        print_model_recommendations()
        return

    print(f"✅ Found {len(models)} installed models:")
    for model in models:
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_mb = size / (1024 * 1024) if size > 0 else 0
        print(f"   📦 {name} ({size_mb:.1f} MB)")

    # 3. Test model generation
    print("\n3️⃣ Testing model generation...")
    test_models = []

    # Select models to test
    for model in models:
        model_name = model.get("name", "")
        if model_name:
            # Take only the main part of model name
            base_name = model_name.split(":")[0]
            if base_name not in [m.split(":")[0] for m in test_models]:
                test_models.append(model_name)

    # Limit test count
    test_models = test_models[:3]

    success_count = 0
    for model_name in test_models:
        print(f"\n🧪 Testing model: {model_name}")
        if test_model_generation(model_name, base_url):
            success_count += 1

    # 4. Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {success_count}/{len(test_models)} models tested successfully")

    if success_count > 0:
        print("\n🎉 Ollama configured correctly! Ready to run simulation:")
        print("   python examples/ollama_simulation.py")
        print("\n💡 More configuration options:")
        print("   python examples/ollama_configs.py")
    else:
        print("\n⚠️  All model tests failed")
        print("Please check that models are correctly installed")
        print_model_recommendations()

if __name__ == "__main__":
    main()