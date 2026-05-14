#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Order Limit Configuration Module

Provides configuration classes and utility functions for adaptive order limits.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AdaptiveLimitsConfig:
    """Adaptive order limit configuration"""
    # Whether adaptive limits are enabled
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
    alpha: float = 1.0      # Std dev coefficient for min limit
    beta: float = 2.0       # Std dev coefficient for max limit

    # Forecast model method parameters
    forecast_horizon: int = 3  # Forecast periods
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'base_min_order': self.base_min_order,
            'base_max_order': self.base_max_order,
            'method': self.method,
            'min_ratio': self.min_ratio,
            'max_ratio': self.max_ratio,
            'window_size': self.window_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'forecast_horizon': self.forecast_horizon,
            'forecast_weight': self.forecast_weight,
            'adaptation_rate': self.adaptation_rate,
            'min_absolute_limit': self.min_absolute_limit,
            'max_absolute_limit': self.max_absolute_limit,
            'role_adjustment': self.role_adjustment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveLimitsConfig':
        """Create config from dictionary"""
        config = cls()
        
        if 'enabled' in data:
            config.enabled = data['enabled']
        if 'base_min_order' in data:
            config.base_min_order = data['base_min_order']
        if 'base_max_order' in data:
            config.base_max_order = data['base_max_order']
        if 'method' in data:
            config.method = data['method']
        if 'min_ratio' in data:
            config.min_ratio = data['min_ratio']
        if 'max_ratio' in data:
            config.max_ratio = data['max_ratio']
        if 'window_size' in data:
            config.window_size = data['window_size']
        if 'alpha' in data:
            config.alpha = data['alpha']
        if 'beta' in data:
            config.beta = data['beta']
        if 'forecast_horizon' in data:
            config.forecast_horizon = data['forecast_horizon']
        if 'forecast_weight' in data:
            config.forecast_weight = data['forecast_weight']
        if 'adaptation_rate' in data:
            config.adaptation_rate = data['adaptation_rate']
        if 'min_absolute_limit' in data:
            config.min_absolute_limit = data['min_absolute_limit']
        if 'max_absolute_limit' in data:
            config.max_absolute_limit = data['max_absolute_limit']
        if 'role_adjustment' in data:
            config.role_adjustment = data['role_adjustment']
        
        return config


def get_default_adaptive_limits_config() -> AdaptiveLimitsConfig:
    """Get default adaptive limits configuration"""
    return AdaptiveLimitsConfig()