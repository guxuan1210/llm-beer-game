from typing import Dict, Any, Optional
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llm_agent import LLMAgent
from agents.mindmap_templates import MindmapTemplate, RoleSpecificTemplates, MindmapFormatter
from config.game_config import GameConfig


class MindmapDecisionAgent(LLMAgent):
    """Mind map-based decision agent

    In information sharing mode, each participant incorporates shared information
    into the decision mind map, then makes a decision with explanation based on the mind map.
    """

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 llm_client: Any,
                 agent_id: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 300):
        """
        Initialize mind map decision agent

        Args:
            role: Agent role
            config: Game configuration
            llm_client: LLM client
            agent_id: Agent ID
            temperature: LLM temperature parameter
            max_tokens: Maximum token count (increased to support mind map generation)
        """
        # If agent_id is None, use default value
        if agent_id is None:
            agent_id = f"mindmap_{role}"

        super().__init__(role, config, llm_client, agent_id, temperature, max_tokens)

        # Mind map related
        self.current_mindmap = None
        self.mindmap_history = []

        self.logger = logging.getLogger(f"MindmapAgent_{role}")

    def make_decision(self) -> int:
        """
        Override decision method, using mind map decision flow

        Returns:
            Order quantity
        """
        if self.config.simulation.information_sharing:
            # In information sharing mode, use mind map decision
            return self.make_decision_with_mindmap()
        else:
            # In non-info-sharing mode, use traditional LLM decision
            return super().make_decision()

    def generate_decision_mindmap(self, context: Dict[str, Any]) -> str:
        """
        Generate personal decision mind map, incorporating shared information

        Args:
            context: Decision context information

        Returns:
            Mind map Markdown text
        """
        # Incorporate shared information into context
        if self.config.simulation.information_sharing and hasattr(self, 'shared_info'):
            context['shared_info'] = self.shared_info

        mindmap_prompt = self._create_mindmap_prompt(context)

        try:
            # Check if llm_client is available
            if self.llm_client is None:
                return self._fallback_mindmap(context)

            response = self.llm_client.generate(
                system_prompt="You are a supply chain decision mind map generation expert. Please generate a structured decision analysis mind map, fully incorporating shared information for global analysis.",
                user_prompt=mindmap_prompt,
                temperature=0.1,
                max_tokens=400
            )

            mindmap = response.strip()
            self.current_mindmap = mindmap
            self.mindmap_history.append({
                'round': context.get('current_round', 0),
                'mindmap': mindmap,
                'shared_info': context.get('shared_info', {})
            })

            return mindmap

        except Exception as e:
            self.logger.error(f"Mind map generation failed: {e}")
            return self._fallback_mindmap(context)

    def _create_mindmap_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create mind map generation prompt

        Args:
            context: Decision context, including state info and shared info

        Returns:
            Mind map prompt
        """
        role_name = {
            'retailer': 'Retailer',
            'wholesaler': 'Wholesaler',
            'distributor': 'Distributor',
            'manufacturer': 'Manufacturer'
        }.get(self.role, self.role)

        # Basic state info
        current_round = context.get('current_round', 0)
        inventory = context.get('inventory', 0)
        backorder = context.get('backorder', 0)
        current_demand = context.get('current_demand', 0)

        prompt = f"""As {role_name}, please generate a decision analysis mind map based on current state and shared information.

**Current State Information** (Round {current_round}):
- Current inventory: {inventory}
- Backorders: {backorder}
- Current demand: {current_demand}
- Incoming shipment: {context.get('incoming_shipment', 0)}
- Holding cost: {context.get('holding_cost', 0)}/unit
- Backorder cost: {context.get('backorder_cost', 0)}/unit
- Delivery time: {context.get('lead_time', self.lead_time)} periods
- Lead time components: Order {context.get('order_lead_time', self.order_lead_time)} periods, Transport {context.get('transport_lead_time', self.transport_lead_time)} periods, Production {context.get('production_lead_time', self.production_lead_time)} periods

**Historical Trends**:
- Recent orders: {context.get('recent_orders', [])}
- Recent inventory: {context.get('recent_inventory', [])}
- Demand history: {context.get('demand_history', [])}
"""

        # Add shared information
        shared_info = context.get('shared_info', {})
        if self.config.simulation.information_sharing and shared_info:
            prompt += f"\n**Shared Information Integration Analysis**:\n"

            # Add other roles' state information
            if 'agents_state' in shared_info:
                prompt += "### Other Participant States:\n"
                for agent_role, info in shared_info['agents_state'].items():
                    if agent_role != self.role:
                        agent_name = {
                            'retailer': 'Retailer',
                            'wholesaler': 'Wholesaler',
                            'distributor': 'Distributor',
                            'manufacturer': 'Manufacturer'
                        }.get(agent_role, agent_role)
                        prompt += f"- {agent_name}: Inventory {info.get('inventory', 'N/A')}, Demand {info.get('demand', 'N/A')}, Backorder {info.get('backorder', 'N/A')}, Recent orders {info.get('recent_orders', [])}\n"

            # Add system-level metrics
            if 'system_metrics' in shared_info:
                metrics = shared_info['system_metrics']
                prompt += f"\n### System Overall Status:\n"
                prompt += f"- System total inventory: {metrics.get('total_inventory', 0)}\n"
                prompt += f"- System total cost: {metrics.get('total_cost', 0):.2f}\n"
                prompt += f"- Bullwhip effect level: {metrics.get('bullwhip_effect', 'unknown')}\n"

            # Add demand trend analysis
            if 'demand_analysis' in shared_info:
                demand_info = shared_info['demand_analysis']
                prompt += f"\n### Demand Trend Analysis:\n"
                prompt += f"- Demand trend: {demand_info.get('trend', 'unknown')}\n"
                prompt += f"- Volatility: {demand_info.get('volatility', 'unknown')}\n"

        # Add coordination guidance
        if context.get('coordinator_guidance'):
            guidance = context['coordinator_guidance']
            # Check if guidance is empty or indicates no intervention needed
            if guidance.strip() and "no intervention needed" not in guidance and guidance.strip() != "":
                prompt += f"\n**Global Coordination Guidance**:\n{guidance}\n"
        elif shared_info.get('coordination_guidance') and self.role in shared_info['coordination_guidance']:
            guidance = shared_info['coordination_guidance'][self.role]
            # Check if guidance is empty or indicates no intervention needed
            if guidance.strip() and "no intervention needed" not in guidance and guidance.strip() != "":
                prompt += f"\n**Coordinator Suggestions**:\n{guidance}\n"

        prompt += f"""

Please generate a Markdown format decision analysis mind map with the following structure:

# {role_name} Decision Analysis Mind Map (Round {current_round})

## Current State Assessment
### Inventory Status Analysis
- Current inventory adequacy assessment (Inventory {inventory}, Backorder {backorder})
- Backorder risk analysis
- Incoming shipment arrival expectations

### Demand Trend Judgment
- Demand trend based on historical data
- Short-term demand forecast
- Demand volatility assessment

## Information Integration Analysis (Based on Shared Information)
### Upstream/Downstream State Awareness
- Supplier inventory status
- Customer demand changes
- Overall supply chain trends

### Coordination Strategy Considerations
- Global coordination guidance interpretation
- Coordination with other participants
- Avoiding bullwhip effect amplification
- System-wide optimization opportunities

## Cost-Benefit Tradeoff
### Holding Cost Analysis
- Excess inventory cost risk
- Optimal inventory level estimation

### Backorder Cost Analysis
- Impact of backorders on service levels
- Cost loss from backorders

## Decision Reasoning Process
### Ordering Strategy Selection
- Optimal strategy based on current state
- Adjustments considering shared information
- Risk control measures
- Coordination effect considerations

### Quantity Determination Logic
- Basic demand satisfaction
- Safety stock considerations
- Cost optimization balance
- Global coordination value embodiment

## Final Decision Recommendation
### Recommended Order Quantity
- Specific number and calculation logic
### Decision Rationale
- Complete logic based on mind map analysis
### Expected Effects
- Impact prediction on individual and system

Please ensure the mind map fully integrates shared information, reflecting the coordination decision advantages in information sharing mode.
"""

        return prompt

    def _fallback_mindmap(self, context: Dict[str, Any]) -> str:
        """
        Fallback mind map

        Args:
            context: Decision context

        Returns:
            Simplified mind map
        """
        role_name = {
            'retailer': 'Retailer',
            'wholesaler': 'Wholesaler',
            'distributor': 'Distributor',
            'manufacturer': 'Manufacturer'
        }.get(self.role, self.role)

        return f"""# {role_name} Decision Analysis Mind Map

## Current State Assessment
- Inventory: {context.get('inventory', 0)}
- Demand: {context.get('current_demand', 0)}
- Backorder: {context.get('backorder', 0)}

## Decision Reasoning
- Decision based on current inventory and demand
- Consider cost balance
- Maintain supply chain stability

## Decision Recommendation
- Adopt robust ordering strategy
- Avoid overreaction
- Focus on long-term benefits
"""

    def _create_fallback_mindmap(self, context: Dict[str, Any]) -> str:
        """
        Create fallback mind map

        Args:
            context: Decision context

        Returns:
            Simplified mind map
        """
        role_name = {
            'retailer': 'Retailer',
            'wholesaler': 'Wholesaler',
            'distributor': 'Distributor',
            'manufacturer': 'Manufacturer'
        }.get(self.role, self.role)

        return f"""# {role_name} Decision Analysis Mind Map

## Current State Assessment
- Inventory: {context.get('inventory', 0)}
- Demand: {context.get('current_demand', 0)}
- Backorder: {context.get('backorder', 0)}

## Decision Reasoning
- Decision based on current inventory and demand
- Consider cost balance
- Maintain supply chain stability

## Decision Recommendation
- Adopt robust ordering strategy
- Avoid overreaction
- Focus on long-term benefits
"""

    def _generate_decision_mindmap(self, state: Dict[str, Any], shared_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate decision mind map

        Args:
            state: Current state information
            shared_info: Shared information (if information sharing is enabled)

        Returns:
            Mind map content
        """
        # Build mind map context
        context = self._build_mindmap_context(state, shared_info)

        # Select appropriate template
        template = self._select_template()

        # Format template
        try:
            mindmap_content = MindmapFormatter.format_template(template, context)
            return mindmap_content
        except Exception as e:
            logging.error(f"Formatting mind map template failed: {e}")
            # Return simplified mind map
            return self._create_fallback_mindmap(context)

    def _build_mindmap_context(self, state: Dict[str, Any], shared_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build mind map context

        Args:
            state: Current state information
            shared_info: Shared information

        Returns:
            Mind map context
        """
        current_inventory = state.get('inventory', 0)
        current_demand = state.get('demand', 0)

        # Basic context
        context = {
            'role': self.role,
            'current_inventory': current_inventory,
            'backorder': state.get('backorder', 0),
            'incoming_shipment': state.get('incoming_shipment', 0),
            'current_demand': current_demand,
            'week': state.get('week', 0)
        }

        # Add historical information (from state)
        demand_history = state.get('demand_history', [])
        if demand_history:
            context['demand_history'] = str(demand_history[-5:])  # Last 5 rounds
            context['inventory_trend'] = 'stable' if len(set(demand_history[-3:])) <= 2 else 'volatile'
        else:
            context['demand_history'] = 'No historical data'
            context['inventory_trend'] = 'unknown'

        order_history = state.get('orders_history', [])
        if order_history:
            context['order_history'] = str(order_history[-5:])  # Last 5 rounds
        else:
            context['order_history'] = 'No historical data'

        # Add decision options
        decision_options = MindmapFormatter.create_decision_options(
            current_inventory, current_demand, self.role
        )
        context.update(decision_options)

        # Add risk assessment
        context.update(self._assess_risks(state))

        # Add shared information
        if shared_info:
            context['shared_info_section'] = self._format_shared_info(shared_info)
            context['has_shared_info'] = True
            # Add system-level metrics
            context.update(self._calculate_system_metrics(shared_info))
        else:
            context['shared_info_section'] = 'Information sharing not enabled'
            context['has_shared_info'] = False
            context['total_inventory'] = 'unknown'
            context['supply_chain_stability'] = 'unknown'
            context['bullwhip_effect'] = 'unknown'

        # Add coordinator guidance (if available)
        context['coordinator_guidance_section'] = self._get_coordinator_guidance()

        # Add decision reasoning
        context['decision_logic'] = self._generate_decision_logic(context)
        context['key_factor_1'] = 'Inventory level and demand matching'
        context['key_factor_2'] = 'Cost-benefit balance'
        context['key_factor_3'] = 'Risk control'

        # Recommended order quantity
        recommended_order = max(0, current_demand - current_inventory + 2)
        context['recommended_order'] = recommended_order
        context['confidence_level'] = 'Medium'
        context['confidence_reason'] = 'Based on current state and historical trends'

        return context

    def _select_template(self) -> str:
        """Select appropriate mind map template

        Returns:
            Mind map template string
        """
        role_templates = {
            'retailer': RoleSpecificTemplates.get_retailer_template(),
            'wholesaler': RoleSpecificTemplates.get_wholesaler_template(),
            'distributor': RoleSpecificTemplates.get_distributor_template(),
            'manufacturer': RoleSpecificTemplates.get_manufacturer_template()
        }

        base_template = MindmapTemplate.get_decision_mindmap_template()
        role_template = role_templates.get(self.role, '')

        if role_template:
            return role_template.replace('{base_template}', base_template)
        else:
            return base_template

    def _assess_risks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks

        Args:
            state: Current state

        Returns:
            Risk assessment result
        """
        inventory = state.get('inventory', 0)
        demand = state.get('demand', 0)
        backorder = state.get('backorder', 0)

        # Stockout risk assessment
        if inventory <= 0 or backorder > 0:
            stockout_risk = 'High'
            stockout_factors = 'Current inventory insufficient, backorders exist'
        elif inventory < demand:
            stockout_risk = 'Medium'
            stockout_factors = 'Inventory below demand'
        else:
            stockout_risk = 'Low'
            stockout_factors = 'Sufficient inventory'

        # Overstock risk assessment
        if inventory > demand * 3:
            overstock_risk = 'High'
            holding_cost_impact = 'Holding costs significantly increased'
        elif inventory > demand * 1.5:
            overstock_risk = 'Medium'
            holding_cost_impact = 'Holding costs moderate'
        else:
            overstock_risk = 'Low'
            holding_cost_impact = 'Holding costs low'

        return {
            'stockout_risk': stockout_risk,
            'stockout_factors': stockout_factors,
            'overstock_risk': overstock_risk,
            'holding_cost_impact': holding_cost_impact
        }

    def _format_shared_info(self, shared_info: Dict[str, Any]) -> str:
        """Format shared information

        Args:
            shared_info: Shared information

        Returns:
            Formatted shared information string
        """
        if not shared_info:
            return 'No shared information available'

        formatted_info = []

        for role, info in shared_info.items():
            if isinstance(info, dict):
                role_info = f"### {role.title()}\n"
                role_info += f"- Inventory: {info.get('inventory', 'unknown')}\n"
                role_info += f"- Demand: {info.get('demand', 'unknown')}\n"
                role_info += f"- Order: {info.get('order', 'unknown')}\n"
                formatted_info.append(role_info)

        return '\n'.join(formatted_info) if formatted_info else 'No valid shared information'

    def _calculate_system_metrics(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system-level metrics

        Args:
            shared_info: Shared information

        Returns:
            System-level metrics
        """
        total_inventory = 0
        inventory_count = 0

        for role, info in shared_info.items():
            if isinstance(info, dict) and 'inventory' in info:
                try:
                    total_inventory += int(info['inventory'])
                    inventory_count += 1
                except (ValueError, TypeError):
                    pass

        # Supply chain stability assessment (simplified)
        if inventory_count > 0:
            avg_inventory = total_inventory / inventory_count
            if avg_inventory > 10:
                stability = 'stable'
                bullwhip = 'low'
            elif avg_inventory > 5:
                stability = 'moderate'
                bullwhip = 'medium'
            else:
                stability = 'unstable'
                bullwhip = 'high'
        else:
            stability = 'unknown'
            bullwhip = 'unknown'

        return {
            'total_inventory': str(total_inventory),
            'supply_chain_stability': stability,
            'bullwhip_effect': bullwhip
        }

    def _get_coordinator_guidance(self) -> str:
        """Get coordinator guidance

        Returns:
            Coordinator guidance information
        """
        # Use parent class coordinator guidance
        return super().get_coordinator_guidance()

    def _generate_decision_logic(self, context: Dict[str, Any]) -> str:
        """
        Generate decision logic

        Args:
            context: Context information

        Returns:
            Decision logic description
        """
        inventory = context.get('current_inventory', 0)
        demand = context.get('current_demand', 0)

        if inventory < demand:
            return f"Current inventory ({inventory}) is below demand ({demand}), need to replenish to avoid stockouts"
        elif inventory > demand * 2:
            return f"Current inventory ({inventory}) is too high, exceeding demand ({demand}) by 2x, should reduce orders"
        else:
             return f"Current inventory ({inventory}) basically matches demand ({demand}), maintain moderate ordering"

    def _parse_decision_response(self, response: str) -> int:
        """
        Parse LLM decision response

        Args:
            response: LLM response text

        Returns:
            Order quantity
        """
        try:
            # Try multiple parsing methods
            lines = response.strip().split('\n')

            for line in lines:
                # Find line containing "order quantity" or "订购数量"
                if ('order quantity' in line.lower() or '订购数量' in line) and ':' in line:
                    quantity_str = line.split(':')[1].strip()
                    # Extract number
                    import re
                    numbers = re.findall(r'\d+', quantity_str)
                    if numbers:
                        quantity = int(numbers[0])
                        if 0 <= quantity <= 1000:
                            return quantity

                # Find pure numeric line
                if line.strip().isdigit():
                    quantity = int(line.strip())
                    if 0 <= quantity <= 1000:
                        return quantity

            # If no clear quantity found, try extracting numbers from entire response
            import re
            numbers = re.findall(r'\b\d+\b', response)
            for num_str in numbers:
                quantity = int(num_str)
                if 0 <= quantity <= 1000:
                    return quantity

        except (ValueError, IndexError, AttributeError) as e:
            self.logger.warning(f"Failed to parse LLM response: {response}, error: {e}")

        # Parse failed, use fallback decision
        self.last_decision_explanation = "LLM parsing failed, using fallback decision"
        return self._fallback_decision()

    def make_decision_with_mindmap(self) -> int:
        """
        Make decision based on mind map

        Returns:
            Order quantity
        """
        try:
            # 1. Prepare decision context
            context = self.get_decision_context()

            # 2. Generate decision mind map
            mindmap = self.generate_decision_mindmap(context)

            # 3. Make decision based on mind map
            decision_prompt = self._create_decision_prompt(context, mindmap)

            # 4. Call LLM to make final decision
            if self.llm_client is None:
                # If no LLM client, use fallback decision
                decision = self._fallback_decision()
                response = f'{{"decision": {decision}, "reason": "No LLM client, using fallback decision"}}'
            else:
                response = self.llm_client.generate(
                    system_prompt=self._create_system_prompt(),
                    user_prompt=decision_prompt,
                    temperature=self.temperature,
                    max_tokens=200
                )

            # 5. Parse decision result
            decision = self._parse_decision_response(response)

            # 5.5. Apply order quantity limits (fix limit application failure in info sharing mode)
            decision = self._apply_order_limits(decision)

            # 6. Record decision explanation
            self.last_decision_explanation = f"Decision based on mind map analysis: {response}"

            # 7. Record decision history
            self.decision_history.append({
                'round': self.current_round,
                'decision': decision,
                'mindmap': mindmap,
                'explanation': response,
                'context': context
            })

            self.logger.info(f"Mind map-based decision complete - Order quantity: {decision}")
            return decision

        except Exception as e:
            self.logger.error(f"Mind map decision failed: {e}")
            # Fallback to traditional decision method
            return super().make_decision()

    def _create_decision_prompt_with_mindmap(self, mindmap: str) -> str:
        """
        Create decision prompt based on mind map

        Args:
            mindmap: Generated mind map text

        Returns:
            Decision prompt
        """
        role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler',
                    'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}.get(self.role, self.role)

        prompt = f"""You have completed a detailed decision analysis mind map, now you need to make the final ordering decision based on this mind map.

## Your Decision Mind Map Analysis:
{mindmap}

## Decision Task
As {role_name}, please determine the optimal order quantity for this round based on the above mind map analysis.

### Decision Requirements:
1. **Quantity Decision**: Return a specific positive integer as the order quantity
2. **Logical Consistency**: Decision must be consistent with the mind map analysis
3. **Concise Explanation**: Explain the decision rationale in 1-2 sentences

### Response Format:
Order Quantity: [specific number]
Decision Rationale: [brief explanation based on mind map]

Please make your decision now:
"""

        return prompt

    def _create_decision_prompt(self, context: Dict[str, Any], mindmap: str) -> str:
        """
        Create decision prompt based on mind map

        Args:
            context: Decision context
            mindmap: Generated mind map

        Returns:
            Decision prompt
        """
        prompt = f"""Based on the following decision analysis mind map, please make a specific ordering decision.

**Decision Analysis Mind Map**:
{mindmap}

**Decision Requirements**:
Please determine the order quantity for this round based on the above mind map analysis.

=== Output Format Requirements ===
You MUST respond in JSON format ONLY:

{{"decision": 15, "reason": "Decision based on mind map analysis"}}

CRITICAL REQUIREMENTS:
- ONLY JSON format is allowed
- Include "decision" field (integer) and "reason" field (brief explanation)
- Reason should be 5-15 words explaining the decision based on mindmap analysis
- NO additional fields in JSON
- NO lengthy explanations outside JSON

CORRECT examples:
{{"decision": 8, "reason": "Mind map shows stable demand"}}
{{"decision": 20, "reason": "Analysis indicates inventory shortage"}}
{{"decision": 5, "reason": "Shared info suggests reducing output"}}

INCORRECT examples (NEVER do this):
12
"I recommend 5"
15|Decision based on mind map analysis
"5 units"
{{"explanation": "need more inventory", "decision": 5}}
"We need 5"
"Based on mindmap analysis, I suggest ordering 10 units"

STRICT RULE: Output ONLY JSON format with decision and brief reason based on mindmap analysis.
"""

        return prompt

    def get_current_mindmap(self) -> Optional[str]:
        """
        Get current mind map

        Returns:
            Current mind map text
        """
        return self.current_mindmap

    def get_mindmap_history(self) -> list:
        """
        Get mind map history

        Returns:
            Mind map history list
        """
        return self.mindmap_history

    def get_decision_explanation_with_mindmap(self) -> Dict[str, Any]:
        """
        Get decision explanation including mind map

        Returns:
            Decision explanation including mind map
        """
        base_explanation = self.get_decision_explanation()
        base_explanation['current_mindmap'] = self.current_mindmap
        base_explanation['mindmap_history'] = self.mindmap_history
        return base_explanation
