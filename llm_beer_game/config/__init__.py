"""Configuration Module

This module contains game configuration management:
- GameConfig: Complete game configuration
- ConfigManager: Configuration manager
- Various predefined configuration functions
"""

from .game_config import (
    GameConfig,
    AgentConfig,
    DemandConfig,
    LLMConfig,
    SimulationConfig,
    CostConfig,
    ConfigManager,
    AgentType,
    DemandPatternType,
    LLMProviderPreset,
    BUILTIN_LLM_PRESETS,
    get_default_config,
    get_info_sharing_config,
    get_rule_based_config,
    get_preset_by_key,
    get_presets_by_provider,
    get_all_preset_keys,
    get_provider_options,
)

__all__ = [
    'GameConfig',
    'AgentConfig',
    'DemandConfig',
    'LLMConfig',
    'SimulationConfig',
    'CostConfig',
    'ConfigManager',
    'AgentType',
    'DemandPatternType',
    'LLMProviderPreset',
    'BUILTIN_LLM_PRESETS',
    'get_default_config',
    'get_info_sharing_config',
    'get_rule_based_config',
    'get_preset_by_key',
    'get_presets_by_provider',
    'get_all_preset_keys',
    'get_provider_options',
]