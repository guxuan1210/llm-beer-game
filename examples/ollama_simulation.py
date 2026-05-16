#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Local LLM Simulation Example

This example shows how to use a local Ollama service to run the LLM Beer Game simulation.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Download required models, e.g.: ollama pull llama2
3. Ollama service running: ollama serve
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_beer_game.config import GameConfig, LLMConfig, AgentConfig, create_standard_config
from llm_beer_game.llm import LLMClientFactory, LLMManager
from llm_beer_game.agents import create_supply_chain
from llm_beer_game.simulation import GameEngine
from llm_beer_game.analysis import BullwhipAnalyzer
from llm_beer_game.visualization import BeerGameVisualizer
from llm_beer_game.utils import setup_logging

def create_ollama_config(model_name: str = "llama2", base_url: str = "http://localhost:11434") -> GameConfig:
    """
    Create game config using Ollama

    Args:
        model_name: Ollama model name (e.g.: llama2, mistral, codellama, etc.)
        base_url: Ollama service address

    Returns:
        Game config object
    """
    # Create base configuration
    config = create_standard_config()

    # Configure Ollama LLM
    ollama_config = LLMConfig(
        provider="ollama",
        model=model_name,
        base_url=base_url,
        temperature=0.3,  # Slightly increase creativity
        max_tokens=200,   # Increase output length
        timeout=60        # Local models may need more time
    )

    # Set Ollama config for all agents
    config.retailer.llm_config = ollama_config
    config.wholesaler.llm_config = ollama_config
    config.distributor.llm_config = ollama_config
    config.manufacturer.llm_config = ollama_config

    return config

def test_ollama_connection(model_name: str = "llama2", base_url: str = "http://localhost:11434") -> bool:
    """
    Test Ollama connection and model availability

    Args:
        model_name: Model name
        base_url: Ollama service address

    Returns:
        Whether connection succeeded
    """
    try:
        client = LLMClientFactory.create_client(
            "ollama",
            model_name=model_name,
            base_url=base_url
        )

        if not client.is_available():
            print(f"❌ Ollama service unavailable ({base_url})")
            print("Please ensure:")
            print("1. Ollama is installed and running: ollama serve")
            print(f"2. Model is downloaded: ollama pull {model_name}")
            return False

        # Test generation
        response = client.generate(
            system_prompt="You are a supply chain management expert.",
            user_prompt="Briefly answer: What is inventory management?",
            temperature=0.1,
            max_tokens=50
        )

        print(f"✅ Ollama connection successful!")
        print(f"Model: {model_name}")
        print(f"Test response: {response[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def run_ollama_simulation():
    """
    Run simulation using Ollama
    """
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    print("🚀 LLM Beer Game - Ollama Local Model Simulation")
    print("=" * 50)

    # Configuration parameters
    model_name = "llama2"  # Can be changed to: mistral, codellama, qwen, etc.
    base_url = "http://localhost:11434"

    # Test connection
    print("\n📡 Testing Ollama connection...")
    if not test_ollama_connection(model_name, base_url):
        print("\n💡 Solutions:")
        print("1. Install Ollama: https://ollama.ai/")
        print(f"2. Download model: ollama pull {model_name}")
        print("3. Start service: ollama serve")
        return

    # Create config
    print("\n⚙️ Creating game config...")
    config = create_ollama_config(model_name, base_url)

    # Create LLM manager
    print("\n🤖 Initializing LLM client...")
    try:
        llm_client = LLMClientFactory.create_client(
            "ollama",
            model_name=model_name,
            base_url=base_url
        )
        llm_manager = LLMManager([llm_client])
    except Exception as e:
        logger.error(f"LLM client initialization failed: {e}")
        return

    # Create supply chain agents
    print("\n👥 Creating supply chain agents...")
    try:
        agents = create_supply_chain(config, llm_client=llm_manager)
        print(f"Created {len(agents)} agents")
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        return

    # Initialize game engine
    print("\n🎮 Initializing game engine...")
    engine = GameEngine(config, agents)

    # Run simulation
    print("\n🏃 Starting simulation (this may take a few minutes, local model inference is slower)...")
    try:
        results = engine.run_simulation()
        print(f"✅ Simulation complete! Ran {len(results.weekly_data)} weeks")
    except Exception as e:
        logger.error(f"Simulation run failed: {e}")
        return

    # Analyze results
    print("\n📊 Analyzing results...")
    analyzer = BullwhipAnalyzer()

    # Calculate total costs
    total_costs = {}
    for agent_name, agent in agents.items():
        total_cost = sum(agent.cost_history)
        total_costs[agent_name] = total_cost
        print(f"{agent_name}: ${total_cost:.2f}")

    total_supply_chain_cost = sum(total_costs.values())
    print(f"\n💰 Total supply chain cost: ${total_supply_chain_cost:.2f}")

    # Bullwhip effect analysis
    try:
        bullwhip_metrics = analyzer.calculate_bullwhip_effect(results)
        print(f"\n📈 Bullwhip effect analysis:")
        print(f"Demand CV: {bullwhip_metrics.demand_cv:.3f}")
        print(f"Order CV: {bullwhip_metrics.order_cv:.3f}")
        print(f"Bullwhip ratio: {bullwhip_metrics.bullwhip_ratio:.3f}")

        if bullwhip_metrics.bullwhip_ratio > 1.5:
            print("⚠️  Significant bullwhip effect detected")
        else:
            print("✅ Bullwhip effect well controlled")

    except Exception as e:
        logger.warning(f"Bullwhip effect analysis failed: {e}")

    # Visualize results
    print("\n📈 Generating visualization charts...")
    try:
        visualizer = BeerGameVisualizer()

        # Save charts
        output_dir = Path("simulation_results")
        output_dir.mkdir(exist_ok=True)

        # Inventory levels chart
        fig1 = visualizer.plot_inventory_levels(results)
        fig1.savefig(output_dir / "ollama_inventory_levels.png", dpi=300, bbox_inches='tight')

        # Orders and demand chart
        fig2 = visualizer.plot_orders_and_demand(results)
        fig2.savefig(output_dir / "ollama_orders_demand.png", dpi=300, bbox_inches='tight')

        # Cost analysis chart
        fig3 = visualizer.plot_costs(results)
        fig3.savefig(output_dir / "ollama_costs.png", dpi=300, bbox_inches='tight')

        print(f"📁 Charts saved to: {output_dir.absolute()}")

    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

    print("\n🎉 Ollama simulation complete!")
    print("\n💡 Tips:")
    print("- Try different Ollama models (mistral, codellama, qwen, etc.)")
    print("- Adjust temperature parameter to change AI decision randomness")
    print("- Use information sharing mode to reduce bullwhip effect")

if __name__ == "__main__":
    run_ollama_simulation()