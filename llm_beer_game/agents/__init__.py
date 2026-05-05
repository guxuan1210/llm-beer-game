"""Supply chain agent module

This module contains various agent implementations for the beer distribution game:
- BaseAgent: Agent base class
- LLMAgent: LLM-based agent
- RuleBasedAgent: Rule-based agent
- Role-specific agents: RetailerAgent, WholesalerAgent, DistributorAgent, ManufacturerAgent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentState
from config.game_config import GameConfig
from agents.llm_agent import LLMAgent, RuleBasedAgent
from agents.supply_chain_agents import (
    RetailerAgent,
    WholesalerAgent,
    DistributorAgent,
    ManufacturerAgent,
    RuleBasedRetailer,
    RuleBasedWholesaler,
    RuleBasedDistributor,
    RuleBasedManufacturer,
    create_agent,
    create_supply_chain
)

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentState',
    'GameConfig',

    # Agent types
    'LLMAgent',
    'RuleBasedAgent',

    # LLM agents
    'RetailerAgent',
    'WholesalerAgent',
    'DistributorAgent',
    'ManufacturerAgent',

    # Rule-based agents
    'RuleBasedRetailer',
    'RuleBasedWholesaler',
    'RuleBasedDistributor',
    'RuleBasedManufacturer',

    # Factory functions
    'create_agent',
    'create_supply_chain'
]
