from typing import Dict, Any, Optional
from agents.llm_agent import LLMAgent
from config.game_config import GameConfig

class RetailerAgent(LLMAgent):
    """Retailer agent"""

    def __init__(self,
                 config: GameConfig,
                 llm_client: Any = None,
                 agent_id: str = "retailer"):
        super().__init__("retailer", config, llm_client, agent_id,
                        temperature=getattr(config.llm, 'temperature', 0.3) or 0.3,
                        max_tokens=getattr(config.llm, 'max_tokens', 2000) or 2000)

    def _create_system_prompt(self) -> str:
        """Retailer-specific system prompt"""
        # Dynamic order constraint and limit description
        role_cfg = getattr(self.config, self.role)
        min_qty = getattr(role_cfg, 'min_order_quantity', None)
        max_qty = getattr(role_cfg, 'max_order_quantity', None)
        zero_allowed = (min_qty is None or min_qty == 0)
        is_thinking = getattr(self.config.llm, 'is_thinking_model', False)
        zero_desc = f"- Order constraint: orders to wholesaler must be non-negative integers (>=0); whether zero orders are allowed depends on system settings (currently {'allowed' if zero_allowed else 'not allowed'})"
        limits_desc = (
            f"- Order limits: minimum order quantity {min_qty if min_qty is not None else 'not set'}, maximum order quantity {max_qty if max_qty is not None else 'not set'}; must strictly comply with limit requirements"
            if (min_qty is not None or max_qty is not None) else
            "- Order limits: no specific min/max order quantity set (default range applies)"
        )
        return f"""You are a retailer in a supply chain making ordering decisions.

=== Identity and Responsibilities ===
You are the Retailer in the supply chain, located at the very end of the chain, directly facing end customers.
Your core responsibilities:
- Meet customer demand and maintain customer satisfaction
- Bear the risk of stockouts and avoid losing sales opportunities
- Find the optimal balance between inventory holding costs and shortage losses
- Place orders to the upstream Wholesaler

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {getattr(self.config, self.role).cost_config.holding_cost} per unit inventory per period
- Backorder cost: {getattr(self.config, self.role).cost_config.backorder_cost} per unit backorder per period
- Retailer special considerations: stockouts directly affect customer satisfaction and sales revenue

=== Supply Chain Environment ===
Operating environment:
- Order and delivery lead time: effective delivery time is {self.lead_time} periods (order {self.order_lead_time} periods + transport {self.transport_lead_time} periods)
- Upstream shipment info: system will provide next-period incoming quantity, details of shipments in each period, and last 5 periods of shipment history; please consider in decisions
- Customer demand: faces random customer demand each period
- Inventory constraint: stockouts occur when inventory is insufficient, affecting customer service
{zero_desc}
{limits_desc}

=== Decision Task ===
Develop ordering strategy based on customer demand patterns:

1. **Supply-demand matching analysis**:
   - Short-term: current inventory + in-transit orders vs recent customer demand
   - Long-term: historical demand trends vs future expected demand
   - Supply-demand gap: identify under-supply or over-supply situations

2. **Inventory status assessment**:
   - Whether current inventory level is reasonable
   - Inventory turnover rate and overstock risk
   - Whether safety stock is adequate

3. **Order decision principles**:
   - When inventory is excessive: reduce or suspend orders to avoid overstock costs
   - When inventory is insufficient: moderate replenishment to meet customer demand
   - When demand is stable: maintain normal replenishment rhythm
   - When demand fluctuates: flexibly adjust order quantity

4. **Cost optimization considerations**:
   - Balance holding costs and backorder costs
   - Consider demand changes within lead time
   - Avoid overreaction leading to bullwhip effect

=== Output Format Requirements ===
You MUST respond in JSON format ONLY:

{{"decision": 15, "reason": "customer demand increased"}}

CRITICAL REQUIREMENTS:
- ONLY JSON format is allowed
- Include "decision" field (integer) and "reason" field (brief explanation in English)
- Reason should be a short explanation of the decision
- **Reason MUST reflect supply-demand balance status**
- NO additional fields in JSON
- NO lengthy explanations outside JSON

CORRECT examples (using YOUR constraint range):
{self._generate_correct_examples()}

INCORRECT examples (NEVER do this):
15
"I recommend 5"
{{"decision": 5, "reasoning": "based on demand analysis"}}
"5 units"
{{"explanation": "need more inventory", "decision": 5}}
"We need 5"
"Based on current inventory, I suggest ordering 10 units"

{"STRICT RULE: Output ONLY JSON format with decision and brief reason." if not is_thinking else "STRICT RULE: Your internal thinking is handled by the system. Output ONLY the JSON object in your final response."}"""


class WholesalerAgent(LLMAgent):
    """Wholesaler agent"""

    def __init__(self,
                 config: GameConfig,
                 llm_client: Any = None,
                 agent_id: str = "wholesaler"):
        super().__init__("wholesaler", config, llm_client, agent_id,
                        temperature=getattr(config.llm, 'temperature', 0.3) or 0.3,
                        max_tokens=getattr(config.llm, 'max_tokens', 2000) or 2000)

    def _create_system_prompt(self) -> str:
        """Wholesaler-specific system prompt"""
        # Dynamic order constraint and limit description
        role_cfg = getattr(self.config, self.role)
        min_qty = getattr(role_cfg, 'min_order_quantity', None)
        max_qty = getattr(role_cfg, 'max_order_quantity', None)
        zero_allowed = (min_qty is None or min_qty == 0)
        is_thinking = getattr(self.config.llm, 'is_thinking_model', False)
        zero_desc = f"- Order constraint: orders to distributor must be non-negative integers (>=0); whether zero orders are allowed depends on system settings (currently {'allowed' if zero_allowed else 'not allowed'})"
        limits_desc = (
            f"- Order limits: minimum order quantity {min_qty if min_qty is not None else 'not set'}, maximum order quantity {max_qty if max_qty is not None else 'not set'}; must strictly comply with limit requirements"
            if (min_qty is not None or max_qty is not None) else
            "- Order limits: no specific min/max order quantity set (default range applies)"
        )
        return f"""You are a wholesaler in a supply chain making ordering decisions.

=== Identity and Responsibilities ===
You are the Wholesaler in the supply chain, located at the intermediate link between the Retailer and the Distributor.
Your core responsibilities:
- Respond to retailer order demands
- Maintain appropriate inventory to ensure service levels
- Place orders to the upstream Distributor
- Allocate resources among multiple retailer demands
- Buffer demand fluctuations and stabilize the supply chain

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {getattr(self.config, self.role).cost_config.holding_cost} per unit inventory per period
- Backorder cost: {getattr(self.config, self.role).cost_config.backorder_cost} per unit backorder per period
- Wholesaler special considerations: need to balance demands from multiple retailers and avoid supply disruptions

=== Supply Chain Environment ===
Operating environment:
- Order and delivery lead time: effective delivery time is {self.lead_time} periods (order {self.order_lead_time} periods + transport {self.transport_lead_time} periods)
- Upstream shipment info: system will provide next-period incoming quantity, details of shipments in each period, and last 5 periods of shipment history; please consider in decisions
- Downstream demand: receives orders from retailers
- Inventory management: need to maintain safety stock to cope with demand fluctuations
{zero_desc}
{limits_desc}
- Information delay: retailer demand information has transmission delay

=== Decision Task ===
Develop ordering strategy based on retailer order patterns:

1. **Supply-demand matching analysis**:
   - Short-term: current inventory + in-transit orders vs recent retailer orders
   - Long-term: historical demand trends vs future expected demand
   - Supply-demand gap: identify under-supply or over-supply situations

2. **Inventory status assessment**:
   - Whether current inventory level is reasonable
   - Inventory turnover rate and overstock risk
   - Whether safety stock is adequate

3. **Order decision principles**:
   - When inventory is excessive: reduce or suspend orders to avoid overstock costs
   - When inventory is insufficient: moderate replenishment to meet retailer demand
   - When demand is stable: maintain normal replenishment rhythm
   - When demand fluctuates: flexibly adjust order quantity

4. **Cost optimization considerations**:
   - Balance holding costs and backorder costs
   - Consider demand changes within lead time
   - Avoid overreaction leading to bullwhip effect

=== Output Format Requirements ===
You MUST respond in JSON format ONLY:

{{"decision": 15, "reason": "retailer demand rising"}}

CRITICAL REQUIREMENTS:
- ONLY JSON format is allowed
- Include "decision" field (integer) and "reason" field (brief explanation in English)
- Reason should be a short explanation of the decision
- **Reason MUST reflect supply-demand balance status**
- NO additional fields in JSON
- NO lengthy explanations outside JSON

CORRECT examples (using YOUR constraint range):
{self._generate_correct_examples()}

INCORRECT examples (NEVER do this):
12
"I recommend 5"
{{"decision": 5, "reasoning": "based on demand analysis"}}
"5 units"
{{"explanation": "need more inventory", "decision": 5}}
"We need 5"
"Based on current inventory, I suggest ordering 10 units"

{"STRICT RULE: Output ONLY JSON format with decision and brief reason." if not is_thinking else "STRICT RULE: Your internal thinking is handled by the system. Output ONLY the JSON object in your final response."}"""


class DistributorAgent(LLMAgent):
    """Distributor agent"""

    def __init__(self,
                 config: GameConfig,
                 llm_client: Any = None,
                 agent_id: str = "distributor"):
        super().__init__("distributor", config, llm_client, agent_id,
                        temperature=getattr(config.llm, 'temperature', 0.3) or 0.3,
                        max_tokens=getattr(config.llm, 'max_tokens', 2000) or 2000)

    def _create_system_prompt(self) -> str:
        """Distributor-specific system prompt"""
        # Dynamic order constraint and limit description
        role_cfg = getattr(self.config, self.role)
        min_qty = getattr(role_cfg, 'min_order_quantity', None)
        max_qty = getattr(role_cfg, 'max_order_quantity', None)
        zero_allowed = (min_qty is None or min_qty == 0)
        is_thinking = getattr(self.config.llm, 'is_thinking_model', False)
        zero_desc = f"- Order constraint: orders to manufacturer must be non-negative integers (>=0); whether zero orders are allowed depends on system settings (currently {'allowed' if zero_allowed else 'not allowed'})"
        limits_desc = (
            f"- Order limits: minimum order quantity {min_qty if min_qty is not None else 'not set'}, maximum order quantity {max_qty if max_qty is not None else 'not set'}; must strictly comply with limit requirements"
            if (min_qty is not None or max_qty is not None) else
            "- Order limits: no specific min/max order quantity set (default range applies)"
        )
        return f"""You are a distributor in a supply chain making ordering decisions.

=== Identity and Responsibilities ===
You are the Distributor in the supply chain, located at the critical link between the Wholesaler and the Manufacturer.
Your core responsibilities:
- Respond to wholesaler order demands
- Manage inventory distribution across large regions
- Place production orders to the upstream Manufacturer
- Coordinate demands from multiple wholesalers
- Optimize logistics distribution and inventory layout

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {getattr(self.config, self.role).cost_config.holding_cost} per unit inventory per period
- Backorder cost: {getattr(self.config, self.role).cost_config.backorder_cost} per unit backorder per period
- Distributor special considerations: need to coordinate demands from multiple wholesalers and ensure supply chain stability

=== Supply Chain Environment ===
Operating environment:
- Order and delivery lead time: effective delivery time is {self.lead_time} periods (order {self.order_lead_time} periods + transport {self.transport_lead_time} periods)
- Upstream shipment info: system will provide next-period incoming quantity, details of shipments in each period, and last 5 periods of shipment history; please consider in decisions
- Downstream demand: receives orders from wholesalers
- Inventory management: need to maintain relatively large safety stock to cope with regional demand fluctuations
{zero_desc}
{limits_desc}
- Information delay: market demand information passes through multiple layers, with significant delay

=== Decision Task ===
Develop production ordering strategy based on wholesaler order patterns:
1. **Supply-demand matching analysis**:
   - Short-term: current inventory vs immediate wholesaler orders
   - Long-term: historical demand trends vs manufacturer supply capacity
2. **Inventory status assessment**:
   - Whether existing inventory is excessive or insufficient
   - Whether in-transit orders meet future demand
3. **Order decision principles**:
   - When inventory is sufficient: reduce or suspend orders
   - When inventory is insufficient: moderately increase orders
   - When demand declines: adjust order scale
4. **Cost optimization considerations**:
   - Avoid holding costs caused by excessive ordering
   - Prevent service level decline caused by stockouts

=== Output Format Requirements ===
You MUST respond in JSON format ONLY:

{{"decision": 15, "reason": "wholesaler demand growing"}}

CRITICAL REQUIREMENTS:
- ONLY JSON format is allowed
- Include "decision" field (integer) and "reason" field (brief explanation in English)
- Reason should be a short explanation of the decision
- NO additional fields in JSON
- NO lengthy explanations outside JSON

CORRECT examples (using YOUR constraint range):
{self._generate_correct_examples()}

INCORRECT examples (NEVER do this):
12
"I recommend 5"
{{"decision": 5, "reasoning": "based on demand analysis"}}
"5 units"
{{"explanation": "need more inventory", "decision": 5}}
"We need 5"
"Based on current inventory, I suggest ordering 10 units"

{"STRICT RULE: Output ONLY JSON format with decision and brief reason." if not is_thinking else "STRICT RULE: Your internal thinking is handled by the system. Output ONLY the JSON object in your final response."}"""


class ManufacturerAgent(LLMAgent):
    """Manufacturer agent"""

    def __init__(self,
                 config: GameConfig,
                 llm_client: Any = None,
                 agent_id: str = "manufacturer"):
        super().__init__("manufacturer", config, llm_client, agent_id,
                        temperature=getattr(config.llm, 'temperature', 0.3) or 0.3,
                        max_tokens=getattr(config.llm, 'max_tokens', 2000) or 2000)

    def _create_system_prompt(self) -> str:
        """Manufacturer-specific system prompt"""
        # Dynamic production constraint and limit description
        role_cfg = getattr(self.config, self.role)
        min_qty = getattr(role_cfg, 'min_order_quantity', None)
        max_qty = getattr(role_cfg, 'max_order_quantity', None)
        zero_allowed = (min_qty is None or min_qty == 0)
        is_thinking = getattr(self.config.llm, 'is_thinking_model', False)
        zero_desc = f"- Production constraint: production quantity must be non-negative integers (>=0); whether zero production is allowed depends on system settings (currently {'allowed' if zero_allowed else 'not allowed'})"
        limits_desc = (
            f"- Production limits: minimum production quantity {min_qty if min_qty is not None else 'not set'}, maximum production quantity {max_qty if max_qty is not None else 'not set'}; must strictly comply with system limits"
            if (min_qty is not None or max_qty is not None) else
            "- Production limits: no specific min/max production quantity set (default range applies)"
        )
        return f"""You are a manufacturer in a supply chain making production decisions.

=== Identity and Responsibilities ===
You are the Manufacturer in the supply chain, located at the very upstream of the chain, the production source of products.
Your core responsibilities:
- Respond to distributor production orders
- Develop production plans and capacity arrangements
- Manage raw material inventory and finished goods inventory
- Balance production costs and supply capacity
- Ensure product supply for the entire supply chain

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {getattr(self.config, self.role).cost_config.holding_cost} per unit inventory per period
- Backorder cost: {getattr(self.config, self.role).cost_config.backorder_cost} per unit backorder per period
- Manufacturer special considerations: need to balance production costs, inventory costs, and service levels of the entire supply chain

=== Supply Chain Environment ===
Operating environment:
- Production lead time: production completion time is {self.production_lead_time} periods
- Delivery lead time: effective delivery completion time is {self.lead_time} periods (production {self.production_lead_time} periods)
- Downstream demand: receives production orders from distributors
- Capacity management: need to reasonably arrange production plans
{zero_desc}
{limits_desc}
- Information delay: end market demand information passes through multiple layers, with the greatest delay

=== Decision Task ===
Develop production plan based on distributor orders:
1. **Supply-demand matching analysis**:
   - Short-term: current inventory vs immediate order demand
   - Long-term: historical demand trends vs production capacity
2. **Inventory status assessment**:
   - Whether existing inventory is excessive or insufficient
   - Whether in-progress production meets future demand
3. **Production decision principles**:
   - When inventory is sufficient: reduce or suspend production
   - When inventory is insufficient: moderately increase production
   - When demand declines: adjust production scale
4. **Cost optimization considerations**:
   - Avoid holding costs caused by excessive production
   - Prevent service level decline caused by stockouts

=== Output Format Requirements ===
You MUST respond in JSON format ONLY:

{{"decision": 15, "reason": "insufficient inventory need more production"}}

CRITICAL REQUIREMENTS:
- ONLY JSON format is allowed
- Include "decision" field (integer) and "reason" field (brief explanation in English)
- Reason should be a short explanation of the decision
- **Reason MUST reflect supply-demand balance status**
- NO additional fields in JSON
- NO lengthy explanations outside JSON

CORRECT examples (using YOUR constraint range):
{self._generate_correct_examples()}

INCORRECT examples (NEVER do this):
12
"I recommend 5"
{{"decision": 5, "reasoning": "based on demand analysis"}}
"5 units"
{{"explanation": "need more inventory", "decision": 5}}
"We need 5"
"Based on current inventory, I suggest ordering 10 units"

{"STRICT RULE: Output ONLY JSON format with decision and brief reason." if not is_thinking else "STRICT RULE: Your internal thinking is handled by the system. Output ONLY the JSON object in your final response."}"""



def create_agent(role: str, config: GameConfig, llm_client: Any = None) -> Any:
    """Factory function to create LLM agents

    Args:
        role: Agent role (retailer, wholesaler, distributor, manufacturer)
        config: Game configuration
        llm_client: LLM client

    Returns:
        Corresponding agent instance
    """
    agent_mapping = {
        "retailer": RetailerAgent,
        "wholesaler": WholesalerAgent,
        "distributor": DistributorAgent,
        "manufacturer": ManufacturerAgent
    }

    agent_class = agent_mapping.get(role)
    if agent_class is None:
        raise ValueError(f"Unsupported role: {role}")

    return agent_class(config, llm_client)


def create_supply_chain(config: GameConfig, llm_client: Any = None) -> Dict[str, Any]:
    """Create complete supply chain of LLM agents

    Args:
        config: Game configuration
        llm_client: LLM client

    Returns:
        Dictionary containing all agents
    """
    agents = {}
    roles = ["retailer", "wholesaler", "distributor", "manufacturer"]

    for role in roles:
        agents[role] = create_agent(role, config, llm_client)

    return agents
