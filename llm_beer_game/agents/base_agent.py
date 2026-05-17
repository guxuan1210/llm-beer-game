from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

# Import correct configuration class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.game_config import GameConfig, AgentConfig

@dataclass
class AgentState:
    """Agent state"""
    inventory: int = 6  # Current inventory
    incoming_shipment: int = 0  # Incoming shipment
    backorder: int = 0  # Backorders
    total_cost: float = 0.0  # Total cost
    round_cost: float = 0.0  # Current round cost
    orders_history: List[int] = field(default_factory=list)  # Order history
    inventory_history: List[int] = field(default_factory=list)  # Inventory history
    cost_history: List[float] = field(default_factory=list)  # Cost history
    demand_history: List[int] = field(default_factory=list)  # Demand history
    shipment_history: List[int] = field(default_factory=list)  # Shipment history

class BaseAgent(ABC):
    """Supply chain agent base class, defining basic interfaces and common functionality for all agents"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 llm_client: Optional[Any] = None,
                 agent_id: str = None):
        """
        Initialize agent, set up basic properties

        Args:
            role: Agent role (retailer, wholesaler, distributor, manufacturer)
            config: Game configuration
            llm_client: LLM client
            agent_id: Agent ID
        """
        self.role = role
        self.agent_id = agent_id or f"{role}_agent"
        self.config = config
        self.llm_client = llm_client

        # Calculate effective lead time based on role: order lead time + transport/production lead time
        if role == 'manufacturer':
            order_lt = 0  # Manufacturer does not count order lead time separately to avoid double-counting with production lead time
            transport_lt = 0
            production_lt = int(getattr(config.simulation, "manufacturer_lead_time", getattr(config.simulation, "lead_time", 2)))
        else:
            order_lt = int(getattr(config.simulation, f"{role}_lead_time", getattr(config.simulation, "lead_time", 2)))
            transport_lt = int(getattr(config, f"{role}_transport_lead_time", 0))
            production_lt = 0

        # Save components for logging and UI display
        self.order_lead_time = order_lt
        self.transport_lead_time = transport_lt
        self.production_lead_time = production_lt

        # Effective delivery time
        self.lead_time = max(1, self.order_lead_time + self.transport_lead_time + self.production_lead_time)

        # Get role-specific configuration
        agent_config = getattr(config, role)

        # Initialize state
        self.state = AgentState(inventory=agent_config.initial_inventory)

        # Order and shipment pipeline (considering delivery time)
        # Prefer AgentConfig.initial_in_transit to initialize in-transit pipeline, aligned with effective lead time length
        init_transit = getattr(agent_config, "initial_in_transit", None)
        if isinstance(init_transit, list) and len(init_transit) > 0:
            normalized = [max(0, int(x)) for x in init_transit]
            if len(normalized) >= self.lead_time:
                self.shipment_pipeline = normalized[:self.lead_time]
            else:
                self.shipment_pipeline = normalized + [0] * (self.lead_time - len(normalized))
        else:
            min_order = getattr(agent_config, "min_order_quantity", None)
            if isinstance(min_order, int) and min_order > 0:
                # Auto-fill: when initial in-transit is not configured, fill with min order quantity up to effective lead time length
                self.shipment_pipeline = [min_order] * self.lead_time
            else:
                self.shipment_pipeline = [0] * self.lead_time
        self.current_round = 1
        self.current_demand = 0
        self.current_order = 0

        # Information sharing related
        self.shared_info: Dict[str, Any] = {}

        # Logging
        self.logger = logging.getLogger(f"Agent.{self.role}")

    def reset(self):
        """Reset agent state for a new game start"""
        # Recalculate effective lead time on each reset (to avoid externally modified config not taking effect)
        role = self.role
        if role == 'manufacturer':
            order_lt = 0
            transport_lt = 0
            production_lt = int(getattr(self.config.simulation, "manufacturer_lead_time", getattr(self.config.simulation, "lead_time", 2)))
        else:
            order_lt = int(getattr(self.config.simulation, f"{role}_lead_time", getattr(self.config.simulation, "lead_time", 2)))
            transport_lt = int(getattr(self.config, f"{role}_transport_lead_time", 0))
            production_lt = 0
        self.order_lead_time = order_lt
        self.transport_lead_time = transport_lt
        self.production_lead_time = production_lt
        self.lead_time = max(1, self.order_lead_time + self.transport_lead_time + self.production_lead_time)

        agent_config = getattr(self.config, self.role)
        self.state = AgentState(inventory=agent_config.initial_inventory)
        # Also initialize in-transit pipeline from initial_in_transit on reset
        init_transit = getattr(agent_config, "initial_in_transit", None)
        if isinstance(init_transit, list) and len(init_transit) > 0:
            normalized = [max(0, int(x)) for x in init_transit]
            if len(normalized) >= self.lead_time:
                self.shipment_pipeline = normalized[:self.lead_time]
            else:
                self.shipment_pipeline = normalized + [0] * (self.lead_time - len(normalized))
        else:
            min_order = getattr(agent_config, "min_order_quantity", None)
            if isinstance(min_order, int) and min_order > 0:
                self.shipment_pipeline = [min_order] * self.lead_time
            else:
                self.shipment_pipeline = [0] * self.lead_time
        self.current_round = 1
        self.current_demand = 0
        self.current_order = 0
        self.shared_info = {}

    def update_round(self, round_num: int):
        """Update round information"""
        self.current_round = round_num

    def receive_demand(self, demand: int):
        """Receive downstream demand"""
        self.current_demand = demand
        # Record demand history
        self.state.demand_history.append(demand)
        max_history_len = max(3, 2 * self.lead_time)
        if len(self.state.demand_history) > max_history_len:
            self.state.demand_history.pop(0)

        # Debug log: demand received (simplified)
        # print(f"Received {self.role}: demand={demand}")

        self.logger.debug(f"Round {self.current_round}: Received demand {demand}")

    def receive_shipment(self):
        """Receive upstream shipment"""
        if self.shipment_pipeline:
            shipment = self.shipment_pipeline.pop(0)
            self.state.inventory += shipment
            self.state.incoming_shipment = shipment
            self.logger.debug(f"Round {self.current_round}: Received shipment {shipment}")
            return shipment
        return 0

    def fulfill_demand(self, demand: int) -> int:
        """Fulfill demand and return actual shipment quantity"""
        # First try to fulfill previous backorders
        total_demand = demand + self.state.backorder

        if self.state.inventory >= total_demand:
            # Sufficient inventory, fully fulfill demand
            self.state.inventory -= total_demand
            fulfilled = total_demand
            self.state.backorder = 0
        else:
            # Insufficient inventory, partially fulfill
            fulfilled = self.state.inventory
            self.state.backorder = total_demand - fulfilled
            self.state.inventory = 0

        # Record shipment history (keep full history, no length limit)
        self.state.shipment_history.append(fulfilled)

        self.logger.debug(f"Round {self.current_round}: Fulfilled {fulfilled}, Backorder {self.state.backorder}")
        return fulfilled

    def place_order(self, order_quantity: int):
        """Place order upstream"""
        self.current_order = order_quantity
        # Add order to pipeline (considering delivery time)
        self.shipment_pipeline.append(order_quantity)
        self.state.orders_history.append(order_quantity)
        self.logger.debug(f"Round {self.current_round}: Placed order {order_quantity}")

    def calculate_costs(self) -> float:
        """Calculate current round costs"""
        agent_config = getattr(self.config, self.role)
        holding_cost = self.state.inventory * agent_config.cost_config.holding_cost
        backorder_cost = self.state.backorder * agent_config.cost_config.backorder_cost

        self.state.round_cost = holding_cost + backorder_cost
        self.state.total_cost += self.state.round_cost
        self.state.cost_history.append(self.state.round_cost)

        self.logger.debug(f"Round {self.current_round}: Costs - Holding: {holding_cost:.2f}, Backorder: {backorder_cost:.2f}, Total: {self.state.round_cost:.2f}")
        return self.state.round_cost

    def update_history(self):
        """Update history records"""
        self.state.inventory_history.append(self.state.inventory)
        # Note: order history is updated in place_order, cost history in calculate_costs
        # This ensures all history records are current

    def get_state_info(self) -> Dict[str, Any]:
        """Get agent state information"""
        # Calculate in-transit order information
        total_in_transit = sum(self.shipment_pipeline)  # Total in-transit inventory
        pending_orders = self.shipment_pipeline.copy()  # Pending shipment orders by period
        next_shipment = self.shipment_pipeline[0] if self.shipment_pipeline else 0  # Next period arrival quantity

        return {
            'role': self.role,
            'agent_id': self.agent_id,
            'round': self.current_round,
            'inventory': self.state.inventory,
            'backorder': self.state.backorder,
            'incoming_shipment': self.state.incoming_shipment,
            'current_demand': self.current_demand,
            'current_order': self.current_order,
            'round_cost': self.state.round_cost,
            'total_cost': self.state.total_cost,
            'orders_history': self.state.orders_history.copy(),
            'inventory_history': self.state.inventory_history.copy(),
            'cost_history': self.state.cost_history.copy(),
            'shipment_history': self.state.shipment_history.copy(),  # Shipment history
            'shipment_pipeline': self.shipment_pipeline.copy(),  # In-transit inventory pipeline
            'total_in_transit': total_in_transit,  # Total in-transit inventory
            'pending_orders': pending_orders,  # Pending shipment order details by period
            'next_shipment': next_shipment,  # Next period arrival quantity
            'orders_in_pipeline': len([x for x in self.shipment_pipeline if x > 0]),  # Pipeline segments with orders
            # Lead time components and effective value for UI and log verification
            'lead_time': self.lead_time,
            'order_lead_time': self.order_lead_time,
            'transport_lead_time': self.transport_lead_time,
            'production_lead_time': self.production_lead_time,
        }

    def set_shared_info(self, shared_info: Dict[str, Any]):
        """Set shared information for information sharing mode"""
        self.shared_info = shared_info

    @abstractmethod
    def make_decision(self) -> int:
        """Make ordering decision - subclass must implement"""
        pass

    def get_decision_context(self) -> Dict[str, Any]:
        """Get decision context information"""
        # Calculate in-transit inventory information
        total_in_transit = sum(self.shipment_pipeline)
        next_shipment = self.shipment_pipeline[0] if self.shipment_pipeline else 0
        pending_orders = self.shipment_pipeline.copy()
        orders_in_pipeline = len([x for x in self.shipment_pipeline if x > 0])

        # Dynamic history window: 2 x lead_time, minimum 3 periods
        hist_window = max(3, 2 * self.lead_time)

        context = {
            # Basic state information
            'role': self.role,
            'current_round': self.current_round,
            'inventory': self.state.inventory,
            'backorder': self.state.backorder,
            'current_demand': self.current_demand,
            'incoming_shipment': self.state.incoming_shipment,

            # In-transit inventory information
            'total_in_transit': total_in_transit,
            'next_shipment': next_shipment,
            'pending_orders': pending_orders,
            'orders_in_pipeline': orders_in_pipeline,

            # Historical information (window = 2 x lead_time, floor 3)
            'hist_window': hist_window,
            'recent_orders': self.state.orders_history[-hist_window:] if self.state.orders_history else [],
            'recent_inventory': self.state.inventory_history[-hist_window:] if self.state.inventory_history else [],
            'recent_costs': self.state.cost_history[-hist_window:] if self.state.cost_history else [],
            'demand_history': self.state.demand_history[-hist_window:] if self.state.demand_history else [],
            'shipment_history': self.state.shipment_history[-hist_window:] if self.state.shipment_history else [],

            # Cost information
            'round_cost': self.state.round_cost,
            'total_cost': self.state.total_cost,
            'holding_cost': getattr(self.config, self.role).cost_config.holding_cost,
            'backorder_cost': getattr(self.config, self.role).cost_config.backorder_cost,

            # Lead time information
            'lead_time': self.lead_time,
            'order_lead_time': self.order_lead_time,
            'transport_lead_time': self.transport_lead_time,
            'production_lead_time': self.production_lead_time,
        }

        # If information sharing is enabled, add shared information
        if self.config.simulation.information_sharing and self.shared_info:
            context['shared_info'] = self.shared_info

        return context

    def process_round(self, demand: int) -> int:
        """Process a complete round flow"""
        # 1. Receive demand
        self.receive_demand(demand)

        # 2. Receive shipment
        self.receive_shipment()

        # 3. Fulfill demand
        fulfilled = self.fulfill_demand(demand)

        # 4. Make ordering decision
        order_quantity = self.make_decision()

        # 5. Place order
        self.place_order(order_quantity)

        # 6. Calculate costs
        self.calculate_costs()

        # 7. Update history
        self.update_history()

        return order_quantity

    def __str__(self) -> str:
        return f"{self.role.capitalize()}Agent(inventory={self.state.inventory}, backorder={self.state.backorder}, cost={self.state.total_cost:.2f})"

    def __repr__(self) -> str:
        return self.__str__()
