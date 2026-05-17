from typing import Dict, List, Any, Optional, Callable
import logging
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from config.game_config import GameConfig
from llm.llm_client import LLMClient

@dataclass
class SimulationResult:
    """Simulation result"""
    total_rounds: int
    total_cost: float
    agent_costs: Dict[str, float]
    agent_states: Dict[str, Dict[str, Any]]
    round_history: List[Dict[str, Any]]
    bullwhip_metrics: Dict[str, float]
    simulation_time: float
    config: GameConfig
@dataclass
class DemandPattern:
    """Demand pattern"""
    pattern_type: str  # Support all new distribution types
    base_demand: int = 4
    step_change: int = 0
    step_round: int = 5
    noise_level: float = 0.0
    seasonal_amplitude: float = 0.0
    seasonal_period: int = 12
    custom_demands: List[int] = field(default_factory=list)

    # Random demand parameters (uniform distribution)
    random_min: int = 1
    random_max: int = 10

    # New distribution parameters - use dict storage for dynamic configuration
    distribution_params: Dict[str, Any] = field(default_factory=dict)

    # Internal state (for stateful distributions like Markov chain)
    _internal_state: Dict[str, Any] = field(default_factory=dict)

class GameEngine:
    """Beer distribution game simulation engine"""

    def __init__(self,
                 config: GameConfig,
                 agents: Dict[str, BaseAgent],
                 demand_pattern: Optional[DemandPattern] = None,
                 seed: Optional[int] = None,
                 enable_coordinator: bool = False,
                 llm_client: Optional[LLMClient] = None):
        """
        Initialize the game engine

        Args:
            config: Game configuration
            agents: Agent dictionary {'retailer': agent, 'wholesaler': agent, ...}
            demand_pattern: Demand pattern
            seed: Random seed
            enable_coordinator: Whether to enable coordinator agent
            llm_client: LLM client (for coordinator)
        """
        self.config = config
        self.agents = agents
        self.demand_pattern = demand_pattern or DemandPattern('constant')

        # Set random seed
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

        # Supply chain order (from downstream to upstream)
        self.supply_chain_order = ['retailer', 'wholesaler', 'distributor', 'manufacturer']

        # Simulation state
        self.current_round = 1
        self.round_history = []
        self.is_running = False

        # Logging
        self.logger = logging.getLogger(f"GameEngine")

        self.enable_coordinator = enable_coordinator

        # Validate agents
        self._validate_agents()

        # Reset all agent states to ensure clean state each time a new engine is created
        self.reset()

    def _validate_agents(self):
        """Validate agent configuration"""
        required_roles = set(self.supply_chain_order)
        provided_roles = set(self.agents.keys())

        if not required_roles.issubset(provided_roles):
            missing = required_roles - provided_roles
            raise ValueError(f"Missing agents for roles: {missing}")

        # Validate agent types
        for role, agent in self.agents.items():
            if not isinstance(agent, BaseAgent):
                raise TypeError(f"Agent for role '{role}' must be instance of BaseAgent")

    def reset(self):
        """Reset simulation state"""
        self.current_round = 1
        self.round_history = []
        self.is_running = False

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()

        self.logger.info("Game engine reset")

    def generate_demand(self, round_num: int) -> int:
        """Generate customer demand - supports multiple distribution types"""
        import random
        import math
        import numpy as np

        pattern = self.demand_pattern
        demand = 0

        try:
            # Basic distribution types
            if pattern.pattern_type == 'constant':
                demand = pattern.base_demand

            elif pattern.pattern_type == 'step':
                if round_num >= pattern.step_round:
                    demand = pattern.base_demand + pattern.step_change
                else:
                    demand = pattern.base_demand

            elif pattern.pattern_type == 'random':
                # Uniform distribution
                if hasattr(pattern, 'random_min') and hasattr(pattern, 'random_max'):
                    demand = random.randint(pattern.random_min, pattern.random_max)
                else:
                    demand = max(0, int(random.gauss(pattern.base_demand, pattern.noise_level)))

            elif pattern.pattern_type == 'seasonal':
                seasonal_factor = 1 + pattern.seasonal_amplitude * math.sin(
                    2 * math.pi * round_num / pattern.seasonal_period
                )
                demand = max(0, int(pattern.base_demand * seasonal_factor))

            elif pattern.pattern_type == 'custom':
                if round_num <= len(pattern.custom_demands):
                    demand = pattern.custom_demands[round_num - 1]
                else:
                    demand = pattern.base_demand

            # New common distribution types
            elif pattern.pattern_type == 'normal':
                params = pattern.distribution_params
                mean = params.get('mean', 10.0)
                std = params.get('std', 2.0)
                demand = max(0, int(random.gauss(mean, std)))

            elif pattern.pattern_type == 'poisson':
                params = pattern.distribution_params
                lambda_val = params.get('lambda', 8.0)
                demand = np.random.poisson(lambda_val)

            elif pattern.pattern_type == 'exponential':
                params = pattern.distribution_params
                lambda_val = params.get('lambda', 0.1)
                base = params.get('base', 5.0)
                demand = max(0, int(base + np.random.exponential(1/lambda_val)))

            elif pattern.pattern_type == 'triangular':
                params = pattern.distribution_params
                min_val = params.get('min', 1.0)
                max_val = params.get('max', 20.0)
                mode = params.get('mode', 10.0)
                demand = max(0, int(np.random.triangular(min_val, mode, max_val)))

            elif pattern.pattern_type == 'beta':
                params = pattern.distribution_params
                alpha = params.get('alpha', 2.0)
                beta = params.get('beta', 5.0)
                scale = params.get('scale', 20.0)
                shift = params.get('shift', 1.0)
                demand = max(0, int(shift + scale * np.random.beta(alpha, beta)))

            elif pattern.pattern_type == 'lognormal':
                params = pattern.distribution_params
                mu = params.get('mu', 2.0)
                sigma = params.get('sigma', 0.5)
                demand = max(0, int(np.random.lognormal(mu, sigma)))

            # Mixed distribution types
            elif pattern.pattern_type == 'bimodal':
                demand = self._generate_bimodal_demand(round_num)

            elif pattern.pattern_type == 'seasonal_random':
                demand = self._generate_seasonal_random_demand(round_num)

            elif pattern.pattern_type == 'trend_cyclic':
                demand = self._generate_trend_cyclic_demand(round_num)

            elif pattern.pattern_type == 'multistage':
                demand = self._generate_multistage_demand(round_num)

            elif pattern.pattern_type == 'markov':
                demand = self._generate_markov_demand(round_num)

            # Real market scenario types
            elif pattern.pattern_type == 'promotion':
                demand = self._generate_promotion_demand(round_num)

            elif pattern.pattern_type == 'competition':
                demand = self._generate_competition_demand(round_num)

            elif pattern.pattern_type == 'diffusion':
                demand = self._generate_diffusion_demand(round_num)

            elif pattern.pattern_type == 'inventory_sensitive':
                demand = self._generate_inventory_sensitive_demand(round_num)

            # Unstable demand types
            elif pattern.pattern_type == 'autoregressive':
                demand = self._generate_autoregressive_demand(round_num)

            elif pattern.pattern_type == 'arma':
                demand = self._generate_arma_demand(round_num)

            elif pattern.pattern_type == 'jump_diffusion':
                demand = self._generate_jump_diffusion_demand(round_num)

            elif pattern.pattern_type == 'poisson_jump':
                demand = self._generate_poisson_jump_demand(round_num)

            elif pattern.pattern_type == 'regime_switching':
                demand = self._generate_regime_switching_demand(round_num)

            elif pattern.pattern_type == 'volatility_clustering':
                demand = self._generate_volatility_clustering_demand(round_num)

            else:
                demand = pattern.base_demand

        except Exception as e:
            print(f"⚠️ Demand generation error: {e}, using base demand value")
            demand = pattern.base_demand

        # Ensure demand is a positive integer
        demand = max(1, int(demand))

        # Simplified demand generation log
        self._log_demand_generation(round_num, demand, pattern.pattern_type)

        return demand

    def _log_demand_generation(self, round_num: int, demand: int, pattern_type: str):
        """Log demand generation"""
        pattern = self.demand_pattern

        if pattern_type == 'random' and hasattr(pattern, 'random_min'):
            print(f"🎯 Round {round_num} demand: {demand} (random range: [{pattern.random_min}, {pattern.random_max}])")
        elif pattern_type == 'step':
            phase = "after step" if round_num >= pattern.step_round else "before step"
            print(f"🎯 Round {round_num} demand: {demand} ({phase})")
        elif pattern_type in ['normal', 'poisson', 'exponential', 'triangular', 'beta', 'lognormal']:
            print(f"🎯 Round {round_num} demand: {demand} ({pattern_type} distribution)")
        elif pattern_type in ['bimodal', 'seasonal_random', 'trend_cyclic', 'multistage', 'markov']:
            print(f"🎯 Round {round_num} demand: {demand} ({pattern_type} mixed)")
        elif pattern_type in ['promotion', 'competition', 'diffusion', 'inventory_sensitive']:
            print(f"🎯 Round {round_num} demand: {demand} ({pattern_type} scenario)")
        else:
            print(f"🎯 Round {round_num} demand: {demand}")

    def _generate_bimodal_demand(self, round_num: int) -> int:
        """Generate bimodal distribution demand"""
        import random
        params = self.demand_pattern.distribution_params

        mean1 = params.get('mean1', 5.0)
        std1 = params.get('std1', 1.0)
        mean2 = params.get('mean2', 15.0)
        std2 = params.get('std2', 2.0)
        weight = params.get('weight', 0.6)

        if random.random() < weight:
            return max(0, int(random.gauss(mean1, std1)))
        else:
            return max(0, int(random.gauss(mean2, std2)))

    def _generate_seasonal_random_demand(self, round_num: int) -> int:
        """Generate seasonal + random mixed demand"""
        import random
        import math
        params = self.demand_pattern.distribution_params

        base = params.get('base', 10.0)
        amplitude = params.get('amplitude', 3.0)
        period = params.get('period', 12)
        noise_std = params.get('noise_std', 1.5)

        # Seasonal component
        seasonal = base + amplitude * math.sin(2 * math.pi * round_num / period)
        # Random noise
        noise = random.gauss(0, noise_std)

        return max(0, int(seasonal + noise))

    def _generate_trend_cyclic_demand(self, round_num: int) -> int:
        """Generate trend + cyclic mixed demand"""
        import math
        params = self.demand_pattern.distribution_params

        base = params.get('base', 8.0)
        slope = params.get('slope', 0.1)
        amplitude = params.get('amplitude', 2.0)
        period = params.get('period', 8)

        # Trend component
        trend = base + slope * round_num
        # Cyclic component
        cyclic = amplitude * math.cos(2 * math.pi * round_num / period)

        return max(0, int(trend + cyclic))

    def _generate_multistage_demand(self, round_num: int) -> int:
        """Generate multi-stage mixed demand"""
        import random
        import numpy as np
        params = self.demand_pattern.distribution_params
        stages = params.get('stages', [])

        if not stages:
            return self.demand_pattern.base_demand

        # Determine current stage
        current_week = 0
        for stage in stages:
            current_week += stage.get('duration', 10)
            if round_num <= current_week:
                stage_type = stage.get('type', 'normal')

                if stage_type == 'normal':
                    mean = stage.get('mean', 10)
                    std = stage.get('std', 2)
                    return max(0, int(random.gauss(mean, std)))
                elif stage_type == 'exponential':
                    lambda_val = stage.get('lambda', 0.1)
                    base = stage.get('base', 5)
                    return max(0, int(base + np.random.exponential(1/lambda_val)))
                else:
                    return stage.get('value', 10)

        # If beyond all stages, use the last stage
        last_stage = stages[-1]
        return last_stage.get('value', self.demand_pattern.base_demand)

    def _generate_markov_demand(self, round_num: int) -> int:
        """Generate Markov chain demand"""
        import random
        params = self.demand_pattern.distribution_params
        states = params.get('states', [3, 8, 15])
        transition_matrix = params.get('transition_matrix', [
            [0.7, 0.2, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.3, 0.6]
        ])

        # Initialize state
        if 'current_state' not in self.demand_pattern._internal_state:
            initial_state = params.get('initial_state', 1)
            self.demand_pattern._internal_state['current_state'] = initial_state

        current_state = self.demand_pattern._internal_state['current_state']

        # State transition
        if current_state < len(transition_matrix):
            probabilities = transition_matrix[current_state]
            next_state = random.choices(range(len(states)), weights=probabilities)[0]
            self.demand_pattern._internal_state['current_state'] = next_state
            return int(states[next_state])

        return int(states[0])

    def _generate_promotion_demand(self, round_num: int) -> int:
        """Generate promotion-driven demand"""
        params = self.demand_pattern.distribution_params

        base_demand = params.get('base_demand', 8.0)
        intensity = params.get('intensity', 3.0)
        start_week = params.get('start_week', 10)
        duration = params.get('duration', 3)
        decay_rate = params.get('decay_rate', 0.5)

        if start_week <= round_num < start_week + duration:
            # During promotion
            return int(base_demand * intensity)
        elif round_num >= start_week + duration:
            # Post-promotion decay
            weeks_after = round_num - (start_week + duration)
            decay_factor = decay_rate ** weeks_after
            boost = (intensity - 1) * decay_factor
            return max(0, int(base_demand * (1 + boost)))
        else:
            # Before promotion
            return int(base_demand)

    def _generate_competition_demand(self, round_num: int) -> int:
        """Generate competition-influenced demand"""
        import random
        params = self.demand_pattern.distribution_params

        base_demand = params.get('base_demand', 10.0)
        market_share = params.get('market_share', 0.4)
        elasticity = params.get('elasticity', 1.5)
        competitor_actions = params.get('competitor_actions', [])

        # Calculate competition effect
        competition_effect = 1.0
        for action in competitor_actions:
            if action['week'] == round_num:
                if action['action'] == 'price_cut':
                    competition_effect *= (1 - action['intensity'])
                elif action['action'] == 'promotion':
                    competition_effect *= (1 - action['intensity'] * 0.5)

        # Add random market volatility
        market_volatility = random.gauss(1.0, 0.1)

        demand = base_demand * market_share * competition_effect * market_volatility
        return max(0, int(demand))

    def _generate_diffusion_demand(self, round_num: int) -> int:
        """Generate new product diffusion demand (Bass model)"""
        params = self.demand_pattern.distribution_params

        market_potential = params.get('market_potential', 1000.0)
        p = params.get('innovation_coeff', 0.03)  # Innovation coefficient
        q = params.get('imitation_coeff', 0.38)   # Imitation coefficient

        # Initialize cumulative adopters count
        if 'cumulative_adopters' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['cumulative_adopters'] = 0

        cumulative = self.demand_pattern._internal_state['cumulative_adopters']

        # Bass model formula
        remaining_potential = market_potential - cumulative
        adoption_rate = p + (q * cumulative / market_potential)
        new_adopters = remaining_potential * adoption_rate

        # Update cumulative adopters
        self.demand_pattern._internal_state['cumulative_adopters'] += new_adopters

        return max(0, int(new_adopters))

    def _generate_inventory_sensitive_demand(self, round_num: int) -> int:
        """Generate inventory-sensitive demand"""
        params = self.demand_pattern.distribution_params

        base_demand = params.get('base_demand', 10.0)
        stockout_penalty = params.get('stockout_penalty', 0.3)
        substitution_rate = params.get('substitution_rate', 0.2)

        # Simplified implementation: adjust demand based on historical stockout
        # In practice, current inventory levels should be considered here
        demand_adjustment = 1.0

        # If there is historical stockout record, reduce demand
        try:
            if hasattr(self, 'agents') and 'retailer' in self.agents:
                retailer_agent = self.agents['retailer']
                if hasattr(retailer_agent, 'get_state_info'):
                    retailer_state = retailer_agent.get_state_info()
                    if retailer_state.get('backorder', 0) > 0:
                        demand_adjustment *= (1 - stockout_penalty)
                elif hasattr(retailer_agent, 'state'):
                    # Directly access state attribute
                    if retailer_agent.state.backorder > 0:
                        demand_adjustment *= (1 - stockout_penalty)
        except Exception as e:
            # If agent state access fails, use default adjustment
            self.logger.debug(f"Unable to access retailer state for demand adjustment: {e}")

        return max(0, int(base_demand * demand_adjustment))

    def _generate_autoregressive_demand(self, round_num: int) -> int:
        """Generate autoregressive demand (AR model)"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # AR model parameters
        base_demand = params.get('base_demand', 10.0)
        ar_coeffs = params.get('ar_coeffs', [0.7])  # AR coefficients, default AR(1)
        noise_std = params.get('noise_std', 2.0)    # Noise standard deviation

        # Initialize demand history
        if 'demand_history' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['demand_history'] = [base_demand] * len(ar_coeffs)

        history = self.demand_pattern._internal_state['demand_history']

        # AR model: X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + ε_t
        ar_component = sum(coeff * history[-(i+1)] for i, coeff in enumerate(ar_coeffs))
        noise = np.random.normal(0, noise_std)

        demand = ar_component + noise

        # Update history
        history.append(demand)
        if len(history) > len(ar_coeffs) + 10:  # Keep sufficient history
            history.pop(0)

        return max(1, int(demand))

    def _generate_arma_demand(self, round_num: int) -> int:
        """Generate ARMA demand (autoregressive moving average model)"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # ARMA model parameters
        base_demand = params.get('base_demand', 10.0)
        ar_coeffs = params.get('ar_coeffs', [0.5])     # AR coefficients
        ma_coeffs = params.get('ma_coeffs', [0.3])     # MA coefficients
        noise_std = params.get('noise_std', 2.0)       # Noise standard deviation

        # Initialize state
        if 'demand_history' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['demand_history'] = [base_demand] * max(len(ar_coeffs), len(ma_coeffs))
        if 'noise_history' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['noise_history'] = [0.0] * len(ma_coeffs)

        demand_history = self.demand_pattern._internal_state['demand_history']
        noise_history = self.demand_pattern._internal_state['noise_history']

        # ARMA model: X_t = c + Σφᵢ*X_{t-i} + Σθⱼ*ε_{t-j} + ε_t
        ar_component = sum(coeff * demand_history[-(i+1)] for i, coeff in enumerate(ar_coeffs))
        ma_component = sum(coeff * noise_history[-(i+1)] for i, coeff in enumerate(ma_coeffs))

        current_noise = np.random.normal(0, noise_std)
        demand = base_demand + ar_component + ma_component + current_noise

        # Update history
        demand_history.append(demand)
        noise_history.append(current_noise)

        # Maintain history length
        max_history = max(len(ar_coeffs), len(ma_coeffs)) + 10
        if len(demand_history) > max_history:
            demand_history.pop(0)
        if len(noise_history) > max_history:
            noise_history.pop(0)

        return max(1, int(demand))

    def _generate_jump_diffusion_demand(self, round_num: int) -> int:
        """Generate jump diffusion demand"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # Jump diffusion parameters
        base_demand = params.get('base_demand', 10.0)
        drift = params.get('drift', 0.0)              # Drift rate
        volatility = params.get('volatility', 0.2)    # Volatility
        jump_intensity = params.get('jump_intensity', 0.1)  # Jump intensity
        jump_mean = params.get('jump_mean', 0.0)      # Jump mean
        jump_std = params.get('jump_std', 0.5)        # Jump standard deviation

        # Initialize current demand level
        if 'current_level' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['current_level'] = base_demand

        current_level = self.demand_pattern._internal_state['current_level']

        # Diffusion part (Geometric Brownian Motion)
        dt = 1.0  # Time step
        diffusion = current_level * (drift * dt + volatility * np.sqrt(dt) * np.random.normal())

        # Jump part (Poisson process)
        jump = 0
        if np.random.random() < jump_intensity:
            jump_size = np.random.normal(jump_mean, jump_std)
            jump = current_level * jump_size

        # Update demand level
        new_level = current_level + diffusion + jump
        self.demand_pattern._internal_state['current_level'] = max(1, new_level)

        return max(1, int(new_level))

    def _generate_poisson_jump_demand(self, round_num: int) -> int:
        """Generate Poisson jump demand"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # Poisson jump parameters
        base_demand = params.get('base_demand', 10.0)
        jump_rate = params.get('jump_rate', 0.2)      # Jump rate
        jump_sizes = params.get('jump_sizes', [-5, -2, 3, 8])  # Possible jump sizes
        jump_probs = params.get('jump_probs', [0.2, 0.3, 0.3, 0.2])  # Jump size probabilities

        # Initialize current demand
        if 'current_demand' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['current_demand'] = base_demand

        current_demand = self.demand_pattern._internal_state['current_demand']

        # Check if a jump occurs
        if np.random.random() < jump_rate:
            # Randomly select jump size
            jump_size = np.random.choice(jump_sizes, p=jump_probs)
            current_demand += jump_size

        # Add small random fluctuation
        noise = np.random.normal(0, 1)
        current_demand += noise

        # Update state
        self.demand_pattern._internal_state['current_demand'] = max(1, current_demand)

        return max(1, int(current_demand))

    def _generate_regime_switching_demand(self, round_num: int) -> int:
        """Generate regime switching demand"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # Regime switching parameters
        regimes = params.get('regimes', [
            {'mean': 8, 'std': 2, 'name': 'low'},
            {'mean': 15, 'std': 3, 'name': 'high'}
        ])
        transition_matrix = params.get('transition_matrix', [
            [0.9, 0.1],  # Transition probability from low demand regime
            [0.15, 0.85]  # Transition probability from high demand regime
        ])

        # Initialize current regime
        if 'current_regime' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['current_regime'] = 0

        current_regime = self.demand_pattern._internal_state['current_regime']

        # Regime transition
        transition_probs = transition_matrix[current_regime]
        new_regime = np.random.choice(len(regimes), p=transition_probs)
        self.demand_pattern._internal_state['current_regime'] = new_regime

        # Generate demand based on current regime
        regime_params = regimes[new_regime]
        demand = np.random.normal(regime_params['mean'], regime_params['std'])

        return max(1, int(demand))

    def _generate_volatility_clustering_demand(self, round_num: int) -> int:
        """Generate volatility clustering demand (GARCH type)"""
        import numpy as np

        params = self.demand_pattern.distribution_params

        # GARCH parameters
        base_demand = params.get('base_demand', 10.0)
        alpha = params.get('alpha', 0.1)      # ARCH coefficient
        beta = params.get('beta', 0.8)       # GARCH coefficient
        omega = params.get('omega', 1.0)     # Constant term

        # Initialize state
        if 'variance' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['variance'] = 1.0
        if 'last_shock' not in self.demand_pattern._internal_state:
            self.demand_pattern._internal_state['last_shock'] = 0.0

        last_variance = self.demand_pattern._internal_state['variance']
        last_shock = self.demand_pattern._internal_state['last_shock']

        # GARCH(1,1): σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        current_variance = omega + alpha * (last_shock ** 2) + beta * last_variance
        current_std = np.sqrt(max(0.1, current_variance))  # Ensure standard deviation is positive

        # Generate current shock
        current_shock = np.random.normal(0, current_std)
        demand = base_demand + current_shock

        # Update state
        self.demand_pattern._internal_state['variance'] = current_variance
        self.demand_pattern._internal_state['last_shock'] = current_shock

        return max(1, int(demand))

    def update_shared_information(self):
        """Update shared information (if information sharing is enabled)"""
        if not self.config.simulation.information_sharing:
            return

        # Collect state information from all agents
        shared_info = {}

        # Add retailer demand information for adaptive order quantity limits
        if hasattr(self.config, 'adaptive_limits') and hasattr(self.config.adaptive_limits, 'enabled') and self.config.adaptive_limits.enabled and 'retailer' in self.agents:
            retailer = self.agents['retailer']
            if hasattr(retailer, 'state') and hasattr(retailer.state, 'demand_history'):
                shared_info['retailer_demand'] = {
                    'current': retailer.current_demand if hasattr(retailer, 'current_demand') else 0,
                    'history': retailer.state.demand_history
                }

        for role, agent in self.agents.items():
            # Basic information
            agent_info = {
                'inventory': agent.state.inventory,
                'backorder': agent.state.backorder,
                'demand': agent.current_demand,
                'order': agent.current_order,
                'cost': agent.state.round_cost
            }

            # Add in-transit inventory information
            if hasattr(agent, 'shipment_pipeline'):
                agent_info['total_in_transit'] = sum(agent.shipment_pipeline)
                agent_info['shipment_pipeline'] = agent.shipment_pipeline.copy()

            # Add historical information (if enabled in config)
            if self.config.simulation.shared_demand_history:
                agent_info['demand_history'] = agent.state.demand_history.copy() if hasattr(agent.state, 'demand_history') else []

            if self.config.simulation.shared_inventory_levels:
                agent_info['inventory_history'] = agent.state.inventory_history.copy() if hasattr(agent.state, 'inventory_history') else []

            # Add order history
            agent_info['orders_history'] = agent.state.orders_history.copy() if hasattr(agent.state, 'orders_history') else []

            shared_info[role] = agent_info

        # Distribute to all agents
        for agent in self.agents.values():
            agent.set_shared_info(shared_info)

    def run_single_round(self, customer_demand: int) -> Dict[str, Any]:
        """Run a single round"""
        round_start_time = time.time()

        # Update round information for all agents
        for agent in self.agents.values():
            agent.update_round(self.current_round)

        # Update shared information
        self.update_shared_information()

        # Process orders from downstream to upstream
        current_demand = customer_demand
        round_data = {
            'round': self.current_round,
            'customer_demand': customer_demand,
            'agents': {},
            'orders_flow': [],
            'total_cost': 0.0,
        }

        # Simplified round start message
        print(f"\n🎮 Round {self.current_round} | Customer demand: {customer_demand}")
        sys.stdout.flush()  # Force flush output buffer

        for role in self.supply_chain_order:
            agent = self.agents[role]

            # Record round start state
            start_state = agent.get_state_info()

            # Simplified agent state display
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}[role]

            # Process round
            order_quantity = agent.process_round(current_demand)

            # Record round end state
            end_state = agent.get_state_info()

            # Enhanced display: add in-transit orders and pending shipment information
            in_transit = end_state.get('total_in_transit', 0)  # Total in-transit orders
            next_shipment = end_state.get('next_shipment', 0)  # Next period arrival quantity
            orders_in_pipeline = end_state.get('orders_in_pipeline', 0)  # Number of pipeline segments with orders

            # Build display string
            display_parts = [
                f"Inventory:{end_state['inventory']}",
                f"Backorder:{end_state['backorder']}",
                f"Order:{order_quantity}",
                f"InTransit:{in_transit}",
                f"NextArrival:{next_shipment}" if next_shipment > 0 else "NextArrival:0",
                f"Cost:${end_state['round_cost']:.2f}"
            ]

            print(f"📊 {role_name}: {' | '.join(display_parts)}")
            sys.stdout.flush()  # Force flush output buffer

            # Record full prompts for this round
            if hasattr(agent, '_create_user_prompt') and hasattr(agent, '_create_system_prompt'):
                try:
                    context = agent.get_decision_context()

                    # Get complete system prompt and user prompt
                    system_prompt = agent._create_system_prompt()
                    user_prompt = agent._create_user_prompt(context)

                    # Save full prompts to round_data
                    if 'prompts' not in round_data:
                        round_data['prompts'] = {}
                    round_data['prompts'][role] = {
                        'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'full_prompt': f"System: {system_prompt}\n\nUser: {user_prompt}"
                    }

                except Exception as e:
                    self.logger.warning(f"Unable to record prompt for {role}: {e}")

            # Collect decision explanation (if LLM agent)
            decision_reason = ""
            if hasattr(agent, 'last_decision_reason'):
                decision_reason = getattr(agent, 'last_decision_reason', '')
            decision_explanation = ""
            if hasattr(agent, 'last_decision_explanation'):
                decision_explanation = getattr(agent, 'last_decision_explanation', '')

            # Record agent data
            round_data['agents'][role] = {
                'start_state': start_state,
                'end_state': end_state,
                'demand_received': current_demand,
                'order_placed': order_quantity,
                'round_cost': agent.state.round_cost,
                'decision_explanation': decision_explanation or decision_reason,
                'decision_reason': decision_reason,
                'inventory': end_state.get('inventory', 0),
                'backorder': end_state.get('backorder', 0),
            }

            # Record order flow
            round_data['orders_flow'].append({
                'from': role,
                'to': self._get_upstream_role(role),
                'quantity': order_quantity
            })

            # Accumulate total cost
            round_data['total_cost'] += agent.state.round_cost

            # The downstream order becomes the upstream demand
            current_demand = order_quantity

        # Simplified round summary
        total_cost = sum(round_data['agents'][role]['round_cost'] for role in self.supply_chain_order)
        print(f"💰 Round total cost: ${total_cost:.1f} | Time: {time.time() - round_start_time:.2f}s\n")

        round_data['processing_time'] = time.time() - round_start_time
        return round_data

    def _get_upstream_role(self, role: str) -> Optional[str]:
        """Get upstream role"""
        try:
            index = self.supply_chain_order.index(role)
            if index < len(self.supply_chain_order) - 1:
                return self.supply_chain_order[index + 1]
        except ValueError:
            pass
        return None

    def run_simulation(self,
                      rounds: Optional[int] = None,
                      progress_callback: Optional[Callable[[int, Dict], None]] = None,
                      realtime_callback: Optional[Callable[[Dict], None]] = None) -> SimulationResult:
        """Run full simulation

        Args:
            rounds: Number of simulation rounds (default from config)
            progress_callback: Progress callback function
            realtime_callback: Real-time data callback function (for 3D visualization)

        Returns:
            Simulation result
        """
        start_time = time.time()
        rounds = rounds or self.config.simulation.total_weeks

        self.logger.info(f"Starting simulation: {rounds} rounds")
        self.is_running = True

        # Print simulation start information
        print(f"\n🚀 Starting LLM Beer Game Simulation")
        print(f"{'='*80}")
        sys.stdout.flush()  # Force flush output buffer
        print(f"📋 Simulation Configuration:")
        print(f"   Rounds: {rounds}")
        # Keep global print, also add per-role printing for verification
        # Print effective lead time components per role
        for role in self.supply_chain_order:
            agent = self.agents[role]
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}[role]
            order_lt = getattr(self.config.simulation, f"{role}_lead_time", self.config.simulation.lead_time)
            transport_lt = getattr(self.config, f"{role}_transport_lead_time", 0) if role != 'manufacturer' else 0
            production_lt = getattr(self.config, "manufacturer_lead_time", self.config.simulation.lead_time) if role == 'manufacturer' else 0
            effective_lt = getattr(agent, 'lead_time', order_lt + transport_lt + production_lt)
            print(f"   {role_name} lead time: Order={order_lt}, Transport={transport_lt}, Production={production_lt}, Effective={effective_lt}")
        print(f"   Information sharing: {'Enabled' if self.config.simulation.information_sharing else 'Disabled'}")
        print(f"   Coordinator: Disabled")
        print(f"   Demand pattern: {self.demand_pattern.pattern_type if self.demand_pattern else 'Default'}")

        print(f"\n👥 Participating roles:")
        for role in self.supply_chain_order:
            agent = self.agents[role]
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}[role]
            agent_config = getattr(self.config, role)
            print(f"   {role_name}: Initial inventory={agent_config.initial_inventory}, Holding cost=${agent_config.cost_config.holding_cost}, Backorder cost=${agent_config.cost_config.backorder_cost}")
        print(f"{'='*80}")
        sys.stdout.flush()  # Force flush output buffer

        try:
            for round_num in range(1, rounds + 1):
                self.current_round = round_num

                # Generate customer demand
                customer_demand = self.generate_demand(round_num)

                # Run round
                round_data = self.run_single_round(customer_demand)
                self.round_history.append(round_data)

                # Call progress callback
                if progress_callback:
                    progress_callback(round_num, round_data)

                # Call realtime callback (for 3D visualization)
                if realtime_callback:
                    # Convert data format to match RealtimeSupplyChain3D expected format
                    agents_data = {}
                    for role, agent in self.agents.items():
                        state_info = agent.get_state_info()
                        agents_data[role] = {
                            'inventory': state_info['inventory'],
                            'current_order': state_info['current_order'],
                            'demand': state_info['current_demand'],  # Field name conversion: current_demand -> demand
                            'backorder': state_info['backorder'],
                            'shipment_pipeline': state_info['shipment_pipeline']
                        }

                    realtime_data = {
                        'round': round_num,
                        'customer_demand': customer_demand,  # Add market demand information
                        'agents': agents_data
                    }
                    realtime_callback(realtime_data)

                self.logger.debug(f"Round {round_num} completed, total cost: ${round_data['total_cost']:.2f}")

        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
        finally:
            self.is_running = False

        simulation_time = time.time() - start_time

        # Calculate final results
        result = self._calculate_results(simulation_time)

        # Print simulation end information
        print(f"\n🏁 LLM Beer Game Simulation Complete")
        print(f"{'='*80}")
        print(f"⏱️  Simulation time: {simulation_time:.2f}s")
        print(f"🎯 Rounds completed: {result.total_rounds}/{rounds}")
        sys.stdout.flush()  # Force flush output buffer

        print(f"\n💰 Final Cost Analysis:")
        for role in self.supply_chain_order:
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}[role]
            cost = result.agent_costs[role]
            percentage = (cost / result.total_cost * 100) if result.total_cost > 0 else 0
            print(f"   {role_name}: ${cost:.2f} ({percentage:.1f}%)")
        print(f"   🏆 Total cost: ${result.total_cost:.2f}")
        sys.stdout.flush()  # Force flush output buffer

        print(f"\n📊 Final Inventory Status:")
        for role in self.supply_chain_order:
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}[role]
            state = result.agent_states[role]
            print(f"   {role_name}: Inventory={state['inventory']}, Backorder={state['backorder']}, InTransit={state['total_in_transit']}")
        sys.stdout.flush()  # Force flush output buffer

        if result.bullwhip_metrics:
            print(f"\n🌊 Bullwhip Effect Analysis:")
            for metric, value in result.bullwhip_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value}")
            sys.stdout.flush()  # Force flush output buffer

        print(f"{'='*80}\n")
        sys.stdout.flush()  # Force flush output buffer

        self.logger.info(f"Simulation completed in {simulation_time:.2f}s, total cost: ${result.total_cost:.2f}")
        return result

    def _calculate_results(self, simulation_time: float) -> SimulationResult:
        """Calculate simulation results"""
        # Calculate total cost and per-agent costs
        agent_costs = {}
        total_cost = 0.0

        for role, agent in self.agents.items():
            agent_costs[role] = agent.state.total_cost
            total_cost += agent.state.total_cost

        # Get final states
        agent_states = {}
        for role, agent in self.agents.items():
            agent_states[role] = agent.get_state_info()

        # Calculate bullwhip effect metrics
        bullwhip_metrics = self._calculate_bullwhip_effect()

        return SimulationResult(
            total_rounds=len(self.round_history),
            total_cost=total_cost,
            agent_costs=agent_costs,
            agent_states=agent_states,
            round_history=self.round_history,
            bullwhip_metrics=bullwhip_metrics,
            simulation_time=simulation_time,
            config=self.config,
        )

    def _calculate_bullwhip_effect(self) -> Dict[str, Any]:
        """Calculate bullwhip effect metrics"""
        if len(self.round_history) < 2:
            return {}

        try:
            import numpy as np

            # Extract order data at each level
            orders_data = {}
            for role in self.supply_chain_order:
                orders = []
                for round_data in self.round_history:
                    if role in round_data['agents']:
                        orders.append(round_data['agents'][role]['order_placed'])
                orders_data[role] = orders

            # Calculate coefficient of variation (std/mean)
            cv_ratios = {}
            for role in self.supply_chain_order:
                if role in orders_data and len(orders_data[role]) > 1:
                    orders = np.array(orders_data[role])
                    if np.mean(orders) > 0:
                        cv = np.std(orders) / np.mean(orders)
                        cv_ratios[role] = float(cv)

            # Calculate bullwhip effect ratios
            bullwhip_ratios = {}
            for i in range(len(self.supply_chain_order) - 1):
                downstream = self.supply_chain_order[i]
                upstream = self.supply_chain_order[i + 1]

                if downstream in cv_ratios and upstream in cv_ratios:
                    if cv_ratios[downstream] > 0:
                        ratio = cv_ratios[upstream] / cv_ratios[downstream]
                        bullwhip_ratios[f"{upstream}_vs_{downstream}"] = float(ratio)

            return {
                'coefficient_of_variation': cv_ratios,
                'bullwhip_ratios': bullwhip_ratios,
                'overall_bullwhip': float(bullwhip_ratios.get('manufacturer_vs_retailer', 1.0)) if bullwhip_ratios else 1.0
            }

        except ImportError:
            self.logger.warning("NumPy not available, skipping bullwhip analysis")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to calculate bullwhip effect: {e}")
            return {}

    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'current_round': self.current_round,
            'is_running': self.is_running,
            'total_rounds_completed': len(self.round_history),
            'agents': {role: agent.get_state_info() for role, agent in self.agents.items()}
        }

    def save_results(self, result: SimulationResult, filepath: str, use_timestamp: bool = False):
        """Save simulation results to file"""
        try:
            # If timestamp is needed, modify file path
            if use_timestamp:
                from pathlib import Path
                path_obj = Path(filepath)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_name = f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
                filepath = str(path_obj.parent / new_name)

            # Aggregate lead time components and effective values per role
            agents_lead_times = {}
            try:
                for role, state in result.agent_states.items():
                    agents_lead_times[role] = {
                        'order_lead_time': state.get('order_lead_time'),
                        'transport_lead_time': state.get('transport_lead_time'),
                        'production_lead_time': state.get('production_lead_time'),
                        'effective_lead_time': state.get('lead_time')
                    }
            except Exception as e:
                self.logger.warning(f"Failed to aggregate agents lead times: {e}")

            # Convert to serializable format
            data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_rounds': result.total_rounds,
                    'simulation_time': result.simulation_time,
                    'total_cost': result.total_cost
                },
                'config': {
                    'simulation': {
                        'total_weeks': result.config.simulation.total_weeks,
                        'lead_time': result.config.simulation.lead_time,
                        'information_sharing': result.config.simulation.information_sharing
                    },
                    'agents': {
                        role: {
                            'initial_inventory': getattr(result.config, role).initial_inventory,
                            'holding_cost': getattr(result.config, role).cost_config.holding_cost,
                            'backorder_cost': getattr(result.config, role).cost_config.backorder_cost
                        }
                        for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']
                    },
                    'agents_lead_times': agents_lead_times
                },
                'results': {
                    'agent_costs': result.agent_costs,
                    'agent_states': result.agent_states,
                    'bullwhip_metrics': result.bullwhip_metrics
                },
                'history': result.round_history
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load simulation results from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise
