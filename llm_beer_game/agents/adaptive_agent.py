#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent implementation based on adaptive order quantity limits

This module extends the base agent class with adaptive order quantity limit functionality.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

from agents.base_agent import BaseAgent
from agents.llm_agent import LLMAgent, RuleBasedAgent
from agents.adaptive_limits import AdaptiveLimitsCalculator
from config.game_config import GameConfig


class AdaptiveBaseAgent(BaseAgent):
    """Base agent class with adaptive order quantity limit support"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 llm_client: Optional[Any] = None,
                 agent_id: str = None):
        """Initialize adaptive agent

        Args:
            role: Agent role
            config: Game configuration
            llm_client: LLM client
            agent_id: Agent ID
        """
        super().__init__(role, config, llm_client, agent_id)

        # Create adaptive limits calculator
        self.limits_calculator = AdaptiveLimitsCalculator(config.adaptive_limits)

        # Record limit adjustment information
        self.limits_adjustment_history: List[Dict[str, Any]] = []

    def update_round(self, round_num: int):
        """Update round info, also update adaptive limits"""
        super().update_round(round_num)

        # If retailer, update demand history
        if self.role == 'retailer' and hasattr(self, 'current_demand') and self.current_demand > 0:
            self.limits_calculator.update_demand_history(self.current_demand)

    def _apply_adaptive_order_limits(self, order_quantity: int) -> int:
        """Apply adaptive order quantity limits

        Args:
            order_quantity: Original order quantity

        Returns:
            Adjusted order quantity
        """
        # Calculate adaptive limits for current role
        min_order, max_order = self.limits_calculator.calculate_limits(self.role)

        # Apply limits
        original_quantity = order_quantity
        adjusted_quantity = max(min_order, min(max_order, order_quantity))

        # Record adjustment info
        if original_quantity != adjusted_quantity:
            adjustment_info = {
                'round': self.current_round,
                'original': original_quantity,
                'adjusted': adjusted_quantity,
                'min_limit': min_order,
                'max_limit': max_order,
                'explanation': self.limits_calculator.get_limits_explanation(self.role)
            }
            self.limits_adjustment_history.append(adjustment_info)

            # Output debug info
            print(f"DEBUG: Adaptive limits - {self.role} order adjusted from {original_quantity} to {adjusted_quantity} [range: {min_order}-{max_order}]")

        return adjusted_quantity

    def get_limits_explanation(self) -> str:
        """Get explanation of current limits"""
        return self.limits_calculator.get_limits_explanation(self.role)

    def get_decision_context(self) -> Dict[str, Any]:
        """Get decision context, adding adaptive limit information"""
        context = super().get_decision_context()

        # Add adaptive limit information
        if self.config.adaptive_limits.enabled:
            min_order, max_order = self.limits_calculator.get_current_limits(self.role)
            context['adaptive_limits'] = {
                'enabled': True,
                'min_order': min_order,
                'max_order': max_order,
                'method': self.config.adaptive_limits.method,
                'explanation': self.get_limits_explanation()
            }

        return context


class AdaptiveLLMAgent(LLMAgent, AdaptiveBaseAgent):
    """LLM agent with adaptive order quantity limit support"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 llm_client: Any,
                 agent_id: str = None,
                 temperature: float = 0.1,
                 max_tokens: int = 50):
        """Initialize adaptive LLM agent"""
        # Use super() to trigger MRO: LLMAgent.__init__ -> AdaptiveBaseAgent.__init__ -> BaseAgent.__init__
        super().__init__(role, config, llm_client, agent_id, temperature, max_tokens)
        # Note: limits_calculator and limits_adjustment_history are created in AdaptiveBaseAgent.__init__

    def _apply_order_limits(self, order_quantity: int) -> int:
        """Override order limit method, using adaptive limits"""
        if self.config.adaptive_limits.enabled:
            return self._apply_adaptive_order_limits(order_quantity)
        else:
            # Use original fixed limits
            return super()._apply_order_limits(order_quantity)


class AdaptiveRuleBasedAgent(RuleBasedAgent, AdaptiveBaseAgent):
    """Rule-based agent with adaptive order quantity limit support"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 strategy: str = 'base_stock',
                 agent_id: str = None):
        """Initialize adaptive rule-based agent"""
        # Use super() to trigger MRO: RuleBasedAgent.__init__ -> AdaptiveBaseAgent.__init__ -> BaseAgent.__init__
        super().__init__(role, config, strategy, agent_id)
        # Note: limits_calculator and limits_adjustment_history are created in AdaptiveBaseAgent.__init__

    def _apply_order_limits(self, order_quantity: int) -> int:
        """Override order limit method, using adaptive limits"""
        if self.config.adaptive_limits.enabled:
            return self._apply_adaptive_order_limits(order_quantity)
        else:
            # Use original fixed limits
            return super()._apply_order_limits(order_quantity)
