"""LLM Beer Game Simulation Framework

An LLM-based beer distribution game simulation framework for studying the bullwhip effect
and the impact of information sharing on supply chain performance.

Modules:
- agents: Supply chain agents (Retailer, Wholesaler, Distributor, Manufacturer)
- llm: LLM interface and management
- simulation: Simulation engine and game logic
- analysis: Bullwhip effect analysis and performance metrics
- config: Configuration management and parameter settings
- visualization: Result visualization and web interface
- utils: General utilities and helper functions

Usage:
    from llm_beer_game import GameEngine, create_supply_chain, get_default_config

    config = get_default_config()
    agents = create_supply_chain(config)
    engine = GameEngine(config, agents)
    result = engine.run_simulation()
"""

__version__ = "1.0.0"
__author__ = "LLM Beer Game Team"
__email__ = "contact@llmbeergame.com"

# Import main classes and functions
from .agents import (
    BaseAgent,
    LLMAgent,
    RuleBasedAgent,
    RetailerAgent,
    WholesalerAgent,
    DistributorAgent,
    ManufacturerAgent,
    create_agent,
    create_supply_chain
)

from .llm import (
    LLMClient,
    OpenAIClient,
    AnthropicClient,
    OllamaClient,
    MockLLMClient,
    LLMClientFactory,
    LLMManager
)

from .simulation import (
    GameEngine,
    SimulationResult,
    DemandPattern
)

from .analysis import (
    BullwhipAnalyzer,
    BullwhipMetrics,
    OrderVariabilityAnalysis
)

from .config import (
    GameConfig,
    AgentConfig,
    DemandConfig,
    LLMConfig,
    SimulationConfig,
    ConfigManager,
    AgentType,
    DemandPatternType,
    get_default_config,
    get_info_sharing_config,
    get_rule_based_config
)

from .visualization import (
    BeerGameVisualizer,
    StreamlitApp,
    create_comparison_plot
)

from .utils import (
    setup_logging,
    PerformanceTimer,
    ConfigValidator,
    DataExporter
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Agent-related
    'BaseAgent',
    'LLMAgent',
    'RuleBasedAgent',
    'RetailerAgent',
    'WholesalerAgent',
    'DistributorAgent',
    'ManufacturerAgent',
    'create_agent',
    'create_supply_chain',

    # LLM-related
    'LLMClient',
    'OpenAIClient',
    'AnthropicClient',
    'OllamaClient',
    'MockLLMClient',
    'LLMClientFactory',
    'LLMManager',

    # Simulation-related
    'GameEngine',
    'SimulationResult',
    'DemandPattern',

    # Analysis-related
    'BullwhipAnalyzer',
    'BullwhipMetrics',
    'OrderVariabilityAnalysis',

    # Config-related
    'GameConfig',
    'AgentConfig',
    'DemandConfig',
    'LLMConfig',
    'SimulationConfig',
    'ConfigManager',
    'AgentType',
    'DemandPatternType',
    'get_default_config',
    'get_info_sharing_config',
    'get_rule_based_config',

    # Visualization-related
    'BeerGameVisualizer',
    'StreamlitApp',
    'create_comparison_plot',

    # Utility-related
    'setup_logging',
    'PerformanceTimer',
    'ConfigValidator',
    'DataExporter'
]