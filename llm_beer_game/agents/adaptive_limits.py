#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive order quantity limit mechanism based on retailer end-user demand

This module provides an adaptive order quantity limit calculator that dynamically
adjusts order quantity limit ranges for each supply chain role based on retailer
end-user demand, to reduce the bullwhip effect and improve supply chain responsiveness.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AdaptiveLimitsConfig:
    """Adaptive order quantity limit configuration"""
    # Whether to enable adaptive limits
    enabled: bool = False

    # Base limit range
    base_min_order: int = 3
    base_max_order: int = 8

    # Adaptive method: 'ratio', 'window', 'forecast'
    method: str = 'ratio'

    # Demand ratio method parameters
    min_ratio: float = 0.5  # Minimum order-to-demand ratio
    max_ratio: float = 1.5  # Maximum order-to-demand ratio

    # Sliding window method parameters
    window_size: int = 5    # Sliding window size
    alpha: float = 1.0      # Std dev coefficient for minimum limit
    beta: float = 2.0       # Std dev coefficient for maximum limit

    # Forecast model method parameters
    forecast_horizon: int = 3  # Forecast horizon periods
    forecast_weight: float = 0.7  # Forecast weight

    # General parameters
    adaptation_rate: float = 0.3  # Adaptation rate (0-1)
    min_absolute_limit: int = 1   # Absolute minimum limit
    max_absolute_limit: int = 100  # Absolute maximum limit

    # Role-specific adjustment coefficients
    role_adjustment: Dict[str, float] = field(default_factory=lambda: {
        'retailer': 1.0,
        'wholesaler': 1.1,
        'distributor': 1.2,
        'manufacturer': 1.3
    })


class AdaptiveLimitsCalculator:
    """Adaptive order quantity limit calculator"""

    def __init__(self, config: AdaptiveLimitsConfig):
        """Initialize calculator

        Args:
            config: Adaptive limits configuration
        """
        self.config = config
        self.retailer_demands: List[int] = []
        self.current_demand: int = 0
        self.average_demand: float = 0.0

        # Current limits for each role
        self.current_limits: Dict[str, Tuple[int, int]] = {}

        # Initialize limits for each role to base limits
        for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
            self.current_limits[role] = (
                self.config.base_min_order,
                self.config.base_max_order
            )

    def update_demand_history(self, demand: int):
        """Update demand history

        Args:
            demand: Current round retailer end-user demand
        """
        self.current_demand = demand
        self.retailer_demands.append(demand)

        # Calculate average demand
        if len(self.retailer_demands) > 0:
            self.average_demand = sum(self.retailer_demands) / len(self.retailer_demands)
        else:
            self.average_demand = demand

    def calculate_limits(self, role: str) -> Tuple[int, int]:
        """Calculate adaptive order quantity limits for the specified role

        Args:
            role: Role name ('retailer', 'wholesaler', 'distributor', 'manufacturer')

        Returns:
            (min_order, max_order): Order quantity limit range
        """
        if not self.config.enabled:
            return (self.config.base_min_order, self.config.base_max_order)

        # Get role adjustment coefficient
        role_factor = self.config.role_adjustment.get(role, 1.0)

        # Calculate limits based on selected method
        if self.config.method == 'ratio':
            limits = self._calculate_ratio_based_limits(role_factor)
        elif self.config.method == 'window':
            limits = self._calculate_window_based_limits(role_factor)
        elif self.config.method == 'forecast':
            limits = self._calculate_forecast_based_limits(role_factor)
        else:
            # Default to ratio method
            limits = self._calculate_ratio_based_limits(role_factor)

        # Apply adaptation rate for smooth transition
        current_min, current_max = self.current_limits.get(role, (self.config.base_min_order, self.config.base_max_order))
        new_min, new_max = limits

        smoothed_min = int(current_min * (1 - self.config.adaptation_rate) + new_min * self.config.adaptation_rate)
        smoothed_max = int(current_max * (1 - self.config.adaptation_rate) + new_max * self.config.adaptation_rate)

        # Ensure min does not exceed max
        smoothed_min = min(smoothed_min, smoothed_max)

        # Apply absolute limits
        final_min = max(self.config.min_absolute_limit, smoothed_min)
        final_max = min(self.config.max_absolute_limit, smoothed_max)

        # Update current limits
        self.current_limits[role] = (final_min, final_max)

        return (final_min, final_max)

    def _calculate_ratio_based_limits(self, role_factor: float) -> Tuple[int, int]:
        """Calculate limits based on demand ratio

        Args:
            role_factor: Role adjustment coefficient

        Returns:
            (min_order, max_order): Order quantity limit range
        """
        if self.average_demand == 0:
            return (self.config.base_min_order, self.config.base_max_order)

        # Calculate ratio of current demand to average demand
        demand_ratio = self.current_demand / self.average_demand

        # Apply ratio to calculate limits
        min_order = int(self.config.base_min_order * demand_ratio * self.config.min_ratio * role_factor)
        max_order = int(self.config.base_max_order * demand_ratio * self.config.max_ratio * role_factor)

        return (min_order, max_order)

    def _calculate_window_based_limits(self, role_factor: float) -> Tuple[int, int]:
        """Calculate limits based on sliding window

        Args:
            role_factor: Role adjustment coefficient

        Returns:
            (min_order, max_order): Order quantity limit range
        """
        # If insufficient historical data, use base limits
        if len(self.retailer_demands) < 2:
            return (self.config.base_min_order, self.config.base_max_order)

        # Get recent demand data
        window_size = min(self.config.window_size, len(self.retailer_demands))
        recent_demands = self.retailer_demands[-window_size:]

        # Calculate statistics
        demand_mean = np.mean(recent_demands)
        demand_std = np.std(recent_demands) if len(recent_demands) > 1 else 1.0

        # Calculate limits based on mean and std dev
        min_order = int(max(1, demand_mean - self.config.alpha * demand_std) * role_factor)
        max_order = int((demand_mean + self.config.beta * demand_std) * role_factor)

        return (min_order, max_order)

    def _calculate_forecast_based_limits(self, role_factor: float) -> Tuple[int, int]:
        """Calculate limits based on simple forecast model

        Args:
            role_factor: Role adjustment coefficient

        Returns:
            (min_order, max_order): Order quantity limit range
        """
        # If insufficient historical data, use base limits
        if len(self.retailer_demands) < 3:
            return (self.config.base_min_order, self.config.base_max_order)

        # Simple moving average forecast
        window_size = min(3, len(self.retailer_demands))
        recent_demands = self.retailer_demands[-window_size:]
        forecast = np.mean(recent_demands)

        # Calculate trend
        if len(self.retailer_demands) >= 2:
            trend = self.retailer_demands[-1] - self.retailer_demands[-2]
            # Apply trend for simple forecast
            forecast = forecast + trend * self.config.forecast_horizon * self.config.forecast_weight

        # Calculate limits based on forecast
        min_order = int(max(1, forecast * 0.7) * role_factor)
        max_order = int(forecast * 1.5 * role_factor)

        return (min_order, max_order)

    def get_current_limits(self, role: str) -> Tuple[int, int]:
        """Get current limit range for the role

        Args:
            role: Role name

        Returns:
            (min_order, max_order): Current order quantity limit range
        """
        return self.current_limits.get(role, (self.config.base_min_order, self.config.base_max_order))

    def get_limits_explanation(self, role: str) -> str:
        """Get explanation of limit calculation

        Args:
            role: Role name

        Returns:
            Explanation text
        """
        min_order, max_order = self.current_limits.get(role, (self.config.base_min_order, self.config.base_max_order))

        if not self.config.enabled:
            return f"Using fixed limits: [{min_order}, {max_order}]"

        method_desc = {
            'ratio': 'Demand Ratio Method',
            'window': 'Sliding Window Method',
            'forecast': 'Forecast Model Method'
        }.get(self.config.method, 'Unknown method')

        explanation = f"Using adaptive limits calculated by {method_desc}: [{min_order}, {max_order}]\n"
        explanation += f"Current retailer demand: {self.current_demand}, Average demand: {self.average_demand:.1f}\n"

        if self.config.method == 'ratio':
            demand_ratio = self.current_demand / max(1, self.average_demand)
            explanation += f"Demand ratio: {demand_ratio:.2f}, Role coefficient: {self.config.role_adjustment.get(role, 1.0)}"
        elif self.config.method == 'window':
            if len(self.retailer_demands) >= 2:
                window_size = min(self.config.window_size, len(self.retailer_demands))
                recent_demands = self.retailer_demands[-window_size:]
                demand_mean = np.mean(recent_demands)
                demand_std = np.std(recent_demands)
                explanation += f"Window size: {window_size}, Demand mean: {demand_mean:.1f}, Std dev: {demand_std:.1f}"

        return explanation
