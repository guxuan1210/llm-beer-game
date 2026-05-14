"""Basic Simulation Example

Demonstrates how to use the LLM Beer Game framework for basic simulation.
"""

import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_beer_game import (
    GameEngine,
    create_supply_chain,
    get_default_config,
    BullwhipAnalyzer,
    BeerGameVisualizer,
    LLMManager,
    MockLLMClient,
    setup_logging
)


def run_basic_simulation():
    """Run basic simulation example"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting basic simulation example")

    # Get default config
    config = get_default_config()
    config.name = "Basic Simulation Example"

    # Create LLM manager (using Mock client for demo)
    mock_client = MockLLMClient()
    llm_manager = LLMManager([mock_client])

    # Create supply chain agents
    agents = create_supply_chain(
        config=config,
        llm_client=llm_manager
    )

    print("Supply chain agents created:")
    for agent in agents:
        print(f"- {agent.role}: {type(agent).__name__}")

    # Create simulation engine
    engine = GameEngine(
        config=config,
        agents=agents
    )

    print(f"\nStarting simulation, total weeks: {config.simulation.total_weeks}")
    print(f"Lead time: {config.simulation.lead_time}")
    print(f"Information sharing: {'Enabled' if config.simulation.information_sharing else 'Disabled'}")

    # Run simulation
    result = engine.run_simulation()

    print(f"\nSimulation complete! Actual weeks run: {len(result.demand_history)}")

    # Calculate total costs
    total_costs = {
        'Retailer': sum([h.total_cost for h in result.retailer_history]),
        'Wholesaler': sum([h.total_cost for h in result.wholesaler_history]),
        'Distributor': sum([h.total_cost for h in result.distributor_history]),
        'Manufacturer': sum([h.total_cost for h in result.manufacturer_history])
    }

    print("\n=== Cost Analysis ===")
    for role, cost in total_costs.items():
        print(f"{role}: ${cost:.2f}")

    total_cost = sum(total_costs.values())
    print(f"Total cost: ${total_cost:.2f}")

    # Bullwhip effect analysis
    analyzer = BullwhipAnalyzer()
    bullwhip_metrics = analyzer.analyze_simulation(result)

    print("\n=== Bullwhip Effect Analysis ===")
    print(f"Demand CV: {bullwhip_metrics.demand_cv:.3f}")
    print(f"Retailer CV: {bullwhip_metrics.retailer_cv:.3f}")
    print(f"Wholesaler CV: {bullwhip_metrics.wholesaler_cv:.3f}")
    print(f"Distributor CV: {bullwhip_metrics.distributor_cv:.3f}")
    print(f"Manufacturer CV: {bullwhip_metrics.manufacturer_cv:.3f}")

    print("\n=== Amplification Ratios ===")
    print(f"Retailer: {bullwhip_metrics.retailer_amplification:.2f}")
    print(f"Wholesaler: {bullwhip_metrics.wholesaler_amplification:.2f}")
    print(f"Distributor: {bullwhip_metrics.distributor_amplification:.2f}")
    print(f"Manufacturer: {bullwhip_metrics.manufacturer_amplification:.2f}")

    # Generate visualizations (optional)
    try:
        visualizer = BeerGameVisualizer()

        print("\nGenerating visualization charts...")
        visualizer.plot_inventory_levels(result)
        visualizer.plot_orders_and_demand(result)
        visualizer.plot_costs(result)
        visualizer.plot_bullwhip_effect(bullwhip_metrics)

        print("Visualization charts displayed")
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        print("Make sure matplotlib and related dependencies are installed")

    return result, bullwhip_metrics


if __name__ == "__main__":
    try:
        result, metrics = run_basic_simulation()
        print("\n✅ Basic simulation example completed successfully!")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()