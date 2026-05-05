"""LLM Beer Game Main Program

Main entry point for the LLM beer game simulation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_beer_game.config import (
    ConfigManager, 
    get_default_config, 
    get_info_sharing_config, 
    get_rule_based_config
)
from llm_beer_game.agents import create_supply_chain
from llm_beer_game.llm import LLMManager, MockLLMClient
from llm_beer_game.simulation import GameEngine
from llm_beer_game.analysis import BullwhipAnalyzer
from llm_beer_game.visualization import BeerGameVisualizer
from llm_beer_game.utils import setup_logging, PerformanceTimer, set_random_seed


def run_single_simulation(config_name: str = "default", 
                         output_dir: Optional[str] = None,
                         verbose: bool = True) -> None:
    """Run a single simulation"""
    # Setup logging
    logger = setup_logging("INFO" if verbose else "WARNING")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        if config_name == "default":
            config = get_default_config()
        elif config_name == "info_sharing":
            config = get_info_sharing_config()
        elif config_name == "rule_based":
            config = get_rule_based_config()
        else:
            # Try loading from file
            try:
                config = config_manager.load_config(f"{config_name}.json")
            except FileNotFoundError:
                logger.error(f"Config file not found: {config_name}.json")
                return

        logger.info(f"Using config: {config.name}")

        # Set random seed
        if config.simulation.random_seed:
            set_random_seed(config.simulation.random_seed)
        
        # Create LLM client
        if config.llm.provider == "mock":
            clients = [MockLLMClient()]
        else:
            # Can create other client types based on config
            # Use Mock client as default for now
            clients = [MockLLMClient()]

        # Create LLM manager
        llm_manager = LLMManager(clients)

        # Use config object directly (already correct GameConfig type)
        game_config = config

        # Create supply chain agents
        agents = create_supply_chain(
            config=game_config,
            llm_client=llm_manager
        )

        # Create demand pattern
        from llm_beer_game.simulation.game_engine import DemandPattern
        demand_pattern = DemandPattern(
            pattern_type=config.demand.pattern_type.value,
            base_demand=config.demand.base_demand,
            step_change=config.demand.step_demand - config.demand.base_demand,
            step_round=config.demand.step_week,
            seasonal_amplitude=config.demand.seasonal_amplitude,
            seasonal_period=config.demand.seasonal_period,
            custom_demands=config.demand.custom_demands,
            # Random demand parameters
            random_min=config.demand.random_min,
            random_max=config.demand.random_max
        )
        
        # Create simulation engine
        engine = GameEngine(
            config=game_config,
            agents=agents,
            demand_pattern=demand_pattern,
            seed=config.simulation.random_seed
        )

        # Run simulation
        with PerformanceTimer("Simulation execution"):
            result = engine.run_simulation()

        logger.info(f"Simulation complete, total weeks: {result.total_rounds}")

        # Analyze bullwhip effect
        analyzer = BullwhipAnalyzer()
        bullwhip_metrics = analyzer.analyze_simulation_result(result)

        # Output key metrics
        print("\n=== Simulation Results Summary ===")
        print(f"Config name: {config.name}")
        print(f"Simulation weeks: {config.simulation.total_weeks}")
        print(f"Information sharing: {'Yes' if config.simulation.information_sharing else 'No'}")

        print("\n=== Total Cost ===")
        role_names = {
            'retailer': 'Retailer',
            'wholesaler': 'Wholesaler',
            'distributor': 'Distributor',
            'manufacturer': 'Manufacturer'
        }

        for role, cost in result.agent_costs.items():
            role_name = role_names.get(role, role)
            print(f"{role_name}: ${cost:.2f}")

        print(f"Total: ${result.total_cost:.2f}")

        print("\n=== Bullwhip Effect Analysis ===")
        print(f"Overall bullwhip effect: {bullwhip_metrics.overall_bullwhip_effect:.3f}")

        print("\nCoefficient of variation by tier:")
        for role, cv in bullwhip_metrics.coefficient_of_variation.items():
            role_name = role_names.get(role, role)
            print(f"{role_name}: {cv:.3f}")

        if bullwhip_metrics.amplification_ratios:
            print("\nAmplification ratios:")
            for pair, ratio in bullwhip_metrics.amplification_ratios.items():
                print(f"{pair}: {ratio:.3f}")
        

        
        # Generate visualizations
        if output_dir:
            visualizer = BeerGameVisualizer()
            visualizer.save_all_plots(result, bullwhip_metrics, output_dir)
            logger.info(f"Charts saved to: {output_dir}")

        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Save simulation results (auto-add timestamp)
            engine.save_results(result, str(output_path / "simulation_result.json"), use_timestamp=True)

            # Save bullwhip analysis results to JSON file
            import json
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            bullwhip_data = {
                'coefficient_of_variation': bullwhip_metrics.coefficient_of_variation,
                'variance_ratios': bullwhip_metrics.variance_ratios,
                'amplification_ratios': bullwhip_metrics.amplification_ratios,
                'overall_bullwhip_effect': bullwhip_metrics.overall_bullwhip_effect
            }
            with open(str(output_path / f"bullwhip_analysis_{timestamp}.json"), 'w', encoding='utf-8') as f:
                json.dump(bullwhip_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Simulation execution failed: {e}")
        raise


def run_comparison_study(output_dir: str = "comparison_results") -> None:
    """Run comparison study"""
    logger = setup_logging()
    logger.info("Starting comparison study")

    # Define comparison scenarios
    scenarios = {
        "Standard": get_default_config(),
        "Info Sharing": get_info_sharing_config(),
        "Rule-Based": get_rule_based_config()
    }

    results = {}
    bullwhip_results = {}

    for scenario_name, config in scenarios.items():
        logger.info(f"Running scenario: {scenario_name}")

        # Set random seed for comparability
        set_random_seed(42)

        # Create LLM manager (use Mock for consistency)
        mock_client = MockLLMClient()
        llm_manager = LLMManager([mock_client])

        # Create agents
        agents = create_supply_chain(
            config=config,
            llm_client=llm_manager
        )

        # Run simulation
        engine = GameEngine(
            config=config,
            agents=agents
        )

        result = engine.run_simulation()
        results[scenario_name] = result

        # Analyze bullwhip effect
        analyzer = BullwhipAnalyzer()
        bullwhip_metrics = analyzer.analyze_simulation(result)
        bullwhip_results[scenario_name] = bullwhip_metrics

    # Generate comparison report
    print("\n" + "="*50)
    print("Comparison Study Results")
    print("="*50)

    print("\nScenario comparison:")
    for scenario_name, result in results.items():
        total_cost = (sum([h.total_cost for h in result.retailer_history]) +
                     sum([h.total_cost for h in result.wholesaler_history]) +
                     sum([h.total_cost for h in result.distributor_history]) +
                     sum([h.total_cost for h in result.manufacturer_history]))

        bullwhip = bullwhip_results[scenario_name]
        avg_amplification = (bullwhip.retailer_amplification +
                           bullwhip.wholesaler_amplification +
                           bullwhip.distributor_amplification +
                           bullwhip.manufacturer_amplification) / 4

        print(f"{scenario_name:12} | Total Cost: ${total_cost:8.2f} | Avg Amplification: {avg_amplification:.2f}")

    # Save comparison results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate timestamped unique directory name
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for scenario_name, result in results.items():
        scenario_dir = output_path / f"{scenario_name.replace(' ', '_')}_{timestamp}"
        scenario_dir.mkdir(exist_ok=True)

        # Save results (auto-add timestamp)
        engine = GameEngine(config=scenarios[scenario_name], agents={})
        engine.save_results(result, str(scenario_dir / "result.json"), use_timestamp=True)

        # Save charts
        visualizer = BeerGameVisualizer()
        visualizer.save_all_plots(result, bullwhip_results[scenario_name], str(scenario_dir))

    logger.info(f"Comparison study complete, results saved to: {output_path}")


def create_sample_configs() -> None:
    """Create sample config files"""
    config_manager = ConfigManager()
    config_manager.create_preset_configs()
    print("Sample config files created")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Beer Game Simulation")
    parser.add_argument("command", choices=["run", "compare", "config", "web"],
                       help="Command to execute")
    parser.add_argument("--config", "-c", default="default",
                       help="Config name or file")
    parser.add_argument("--output", "-o", default="results",
                       help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    if args.command == "run":
        run_single_simulation(args.config, args.output, args.verbose)
    elif args.command == "compare":
        run_comparison_study(args.output)
    elif args.command == "config":
        create_sample_configs()
    elif args.command == "web":
        try:
            from llm_beer_game.ui import StreamlitBeerGameApp
            app = StreamlitBeerGameApp()
            app.run_app()
        except ImportError:
            print("Streamlit required to run web UI: pip install streamlit")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()