"""Simulation Engine Module

This module contains the beer distribution game simulation engine:
- GameEngine: Game simulation engine
- SimulationResult: Simulation result data class
- DemandPattern: Demand pattern configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .game_engine import GameEngine, SimulationResult, DemandPattern

__all__ = [
    'GameEngine',
    'SimulationResult', 
    'DemandPattern'
]