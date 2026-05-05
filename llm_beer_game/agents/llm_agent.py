from typing import Dict, Any, Optional
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from config.game_config import GameConfig


class LLMAgent(BaseAgent):
    """LLM-based supply chain agent"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 llm_client: Any,
                 agent_id: str = "",
                 temperature: float = 0.1,  # slightly increase temperature to avoid truncation
                 max_tokens: int = 5000):     # increase token count to ensure complete response
        """
        Initialize LLM agent

        Args:
            role: Agent role
            config: Game configuration
            llm_client: LLM client
            agent_id: Agent ID
            temperature: LLM temperature parameter
            max_tokens: Maximum token count
        """
        super().__init__(role, config, llm_client, agent_id)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Coordinator guidance
        self.coordinator_guidance = None

        # Decision history
        self.decision_history = []
        self.demand_signal_history = []

        # Store last reasoning process
        self.last_reasoning = ""

        # Decision explanation
        self.last_decision_explanation = ""

        # Store last decision reason
        self.last_decision_reason = ""

        # Decision history (including reasons)
        self.decision_reasons_history = []

        # Role descriptions
        self.role_descriptions = {
            'retailer': 'Retailer: Directly faces end customers, needs to meet customer demand',
            'wholesaler': 'Wholesaler: Purchases from distributor, supplies retailer',
            'distributor': 'Distributor: Purchases from manufacturer, supplies wholesaler',
            'manufacturer': 'Manufacturer: Produces products, supplies distributor'
        }

    def _create_system_prompt(self) -> str:
        """Create system prompt"""
        return f"""You are a {self.role} in a supply chain making ordering decisions.

=== Identity and Responsibilities ===
You are the {self.role} in the supply chain, responsible for making ordering decisions to optimize overall supply chain performance.
Your core responsibility is to minimize total costs while meeting downstream demand.

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {getattr(self.config, self.role).cost_config.holding_cost} yuan per unit per period
- Backorder cost: {getattr(self.config, self.role).cost_config.backorder_cost} yuan per unit per period
- Key tradeoff: Backorder cost is {getattr(self.config, self.role).cost_config.backorder_cost/getattr(self.config, self.role).cost_config.holding_cost:.1f}x holding cost, balance carefully

=== Supply Chain Environment ===
Operating environment:
- Lead time: Effective delivery time is {self.lead_time} periods ({('Order '+str(self.order_lead_time)+'+Transport '+str(self.transport_lead_time)) if self.role != 'manufacturer' else ('Production '+str(self.production_lead_time))})
{'' if self.role == 'manufacturer' else '- Upstream shipment info: The system will provide next-period arrival quantities, detailed pending shipments for each period, and the last 5 periods of shipment history. Consider these in your decision.'}
- Information flow: Demand information propagates upstream in stages, with inherent delays
- Order constraints: Must be non-negative integers (>=0); whether zero orders are allowed depends on system settings
- Order limits: If the system has set order limit conditions (min/max quantity or discrete point constraints), you must strictly follow them
- Processing rule: Orders are processed on a First-In-First-Out (FIFO) basis

=== Decision Task ===
You need to make an ordering decision based on current state information:
1. Analyze current inventory, backorders, and in-transit orders
2. Evaluate historical demand trends and cost changes
3. Weigh holding costs against backorder risk
4. Determine the optimal order quantity to minimize total cost

=== Output Format Requirements ===
CRITICAL REQUIREMENT - OUTPUT FORMAT:
You must strictly follow these requirements:

1. Output ONLY a single JSON object, without any other text, explanation, or reasoning process
2. Use exact format: {{"order_decision": <number>, "reasoning": "<text>"}}
3. "order_decision" must be a non-negative integer (0 or greater)
4. "reasoning" should be a brief explanation in English (10-30 characters)

Correct output examples:
{{"order_decision": 5, "reasoning": "Stock low, need replenishment"}}
{{"order_decision": 12, "reasoning": "Demand rising, increase order"}}
{{"order_decision": 0, "reasoning": "Stock sufficient, skip order"}}
{{"order_decision": 8, "reasoning": "Balance cost and risk"}}

Incorrect output examples (PROHIBITED):
- Plain number: 5
- With explanation: I suggest ordering 5 units, because...
- Wrong format: {{"decision": 5}}
- With reasoning text: Based on analysis, {{"order_decision": 5, "reasoning": "need restock"}}

Important reminders:
- Do NOT output any reasoning process or thinking steps
- Do NOT add any extra explanatory text
- Output the JSON object directly, nothing else
- Ensure the JSON format is complete and parseable
        """

    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """Create user prompt"""
        # Basic state info
        inventory = context.get('inventory', 0)
        backorders = context.get('backorder', 0)
        demand = context.get('current_demand', 0)

        prompt = f"""=== Current Status ===
- Current inventory: {inventory} units
- Backorders: {backorders} units
- Current period demand: {demand} units"""

        # In-transit inventory info
        total_in_transit = context.get('total_in_transit', 0)
        next_shipment = context.get('next_shipment', 0)
        pending_orders = context.get('pending_orders', [])
        orders_in_pipeline = context.get('orders_in_pipeline', 0)
        pipeline = context.get('shipment_pipeline', [])

        if self.role == 'manufacturer':
            # Production lead time and breakdown sequence (zero-padded to LT length)
            lt_prod = context.get('lead_time', len(pending_orders) if pending_orders else 0)
            full_seq = [pending_orders[idx] if idx < len(pending_orders) else 0 for idx in range(lt_prod)] if lt_prod > 0 else (pending_orders or [])
            prompt += f"""\n\n=== Production Pipeline Info ===
- Production lead time: {lt_prod} periods
- Total work-in-progress: {total_in_transit} units
- Next period production completion: {next_shipment} units
- Planned production segments: {orders_in_pipeline} periods
- Production completion schedule: {full_seq}
- Production plans within lead time will complete sequentially from period 1 to period {lt_prod}"""
        else:
            prompt += f"""\n\n=== In-Transit Inventory Info ===
- Total in-transit inventory: {total_in_transit} units
- Next period arrival: {next_shipment} units
- Pipeline segments with orders: {orders_in_pipeline} periods
- Pending shipment details by period: {pending_orders}"""

        # Per-period in-transit/production breakdown: show complete pipeline structure
        # Even if nothing in transit/WIP, show the full pipeline structure
        lt = context.get('lead_time', len(pending_orders) if pending_orders else 0)
        if lt > 0:
            # Show complete pipeline structure, including zero-value periods
            breakdown_lines = []
            for idx in range(lt):
                qty = pending_orders[idx] if idx < len(pending_orders) else 0
                if self.role == 'manufacturer':
                    breakdown_lines.append(f"- Period {idx+1} expected completion: {qty} units")
                else:
                    breakdown_lines.append(f"- Period {idx+1} expected arrival: {qty} units")
            prompt += "\n" + "\n".join(breakdown_lines)
            if self.role == 'manufacturer':
                prompt += f"\nNote: Production lead time is {lt} periods. The breakdown above shows completions per period, and each period rotates: after each period completes, the queue advances (period 1 is the most recent completion)."
            else:
                prompt += f"\nNote: The in-transit pipeline shows all in-transit orders for the next {lt} periods (period 1 is the nearest arrival)."
        else:
            if self.role == 'manufacturer':
                prompt += "\n- Per-period production completion breakdown: []"
            else:
                prompt += "\n- Per-period in-transit breakdown: []"

        # Lead time info display
        lt = context.get('lead_time', 0)
        olt = context.get('order_lead_time', 0)
        tlt = context.get('transport_lead_time', 0)
        plt = context.get('production_lead_time', 0)
        prompt += f"""\n\n=== Lead Time Info ===
- Effective delivery time: {lt} periods
- Components: Order {olt} + Transport {tlt} + Production {plt}
- Reminder: Please incorporate lead time components into your ordering decision"""

        # Historical trend info
        recent_orders = context.get('recent_orders', [])
        recent_inventory = context.get('recent_inventory', [])
        recent_costs = context.get('recent_costs', [])
        demand_history = context.get('demand_history', [])
        shipment_history = context.get('shipment_history', [])

        if recent_orders:
            prompt += f"""\n\n=== Historical Trend Info ===
- Last 5 periods order history: {recent_orders}
- Last 5 periods inventory history: {recent_inventory}
- Last 5 periods cost history: {[f'{c:.1f}' for c in recent_costs] if recent_costs else []}
- Last 10 periods demand history: {demand_history}
- Last 5 periods shipment history: {shipment_history}"""

        # Cost analysis info
        round_cost = context.get('round_cost', 0)
        total_cost = context.get('total_cost', 0)
        holding_cost = context.get('holding_cost', 0)
        backorder_cost = context.get('backorder_cost', 0)

        prompt += f"""\n\n=== Cost Analysis ===
- Current period cost: {round_cost:.2f}
- Cumulative total cost: {total_cost:.2f}
- Holding cost rate: {holding_cost}/unit/period
- Backorder cost rate: {backorder_cost}/unit/period"""

        # Cost trend analysis
        if len(recent_costs) >= 3:
            cost_trend = "rising" if recent_costs[-1] > recent_costs[-3] else "falling"
            cost_volatility = max(recent_costs) - min(recent_costs)
            prompt += f"""\n\n=== Cost Volatility Analysis ===
- Cost trend: {cost_trend}
- Cost volatility range: {cost_volatility:.2f}"""

        # Inventory status analysis
        if recent_inventory:
            avg_inventory = sum(recent_inventory) / len(recent_inventory)
            inventory_trend = "increasing" if inventory > avg_inventory else "decreasing"
            prompt += f"""\n\n=== Inventory Status Analysis ===
- Average inventory level: {avg_inventory:.1f} units
- Inventory trend: {inventory_trend}
- Inventory adequacy: {'adequate' if inventory + total_in_transit >= demand * 2 else 'tight'}"""

        # Add coordinator guidance info
        if 'coordinator_guidance' in context and context['coordinator_guidance']:
            guidance = context['coordinator_guidance']
            # Check if guidance is empty or indicates no intervention needed
            if guidance.strip() and "no intervention needed" not in guidance and guidance.strip() != "":
                prompt += f"\n\n=== Coordinator Guidance ===\n{guidance}\n"
                prompt += "Please incorporate the coordinator's guidance in your decision, especially when specific order quantities are provided.\n"

        # If information sharing is enabled, add global info
        if self.config.simulation.information_sharing and 'shared_info' in context:
            shared = context['shared_info']
            prompt += f"\n\n=== GLOBAL INFORMATION SHARING MODE - Supply-Demand Balance Analysis ===\n"
            prompt += f"\nCORE PRINCIPLE: All end-user demand must be met; all participants should maintain supply-demand balance!\n"

            # 1. Retailer total demand info - important shared information
            retailer_info = shared.get('retailer', {})
            retailer_demand_history = retailer_info.get('demand_history', [])
            current_retail_demand = retailer_info.get('demand', context.get('current_demand', 0))

            prompt += f"\n[CORE] Retailer Total Demand Info (All End-User Demand):\n"
            if retailer_demand_history:
                total_retailer_demand = sum(retailer_demand_history)
                avg_retailer_demand = total_retailer_demand / len(retailer_demand_history) if len(retailer_demand_history) > 0 else 0
                recent_retailer_demand = sum(retailer_demand_history[-3:]) if len(retailer_demand_history) >= 3 else total_retailer_demand
                prompt += f"- Retailer historical total demand: {total_retailer_demand} units ({len(retailer_demand_history)} periods total)\n"
                prompt += f"- Retailer average demand: {avg_retailer_demand:.1f} units/period\n"
                prompt += f"- Retailer last 3 periods demand: {recent_retailer_demand} units\n"
            else:
                prompt += f"- Current period retailer demand: {current_retail_demand} units\n"
            prompt += f"- Current period retailer demand: {current_retail_demand} units\n"
            prompt += f"\nIMPORTANT: This is the market's real demand. Total orders across all supply chain participants should stay balanced with it!\n"


            # 2. My total order status analysis - clarify own ordering situation
            my_recent_orders = sum(context.get('recent_orders', [])[-3:]) if context.get('recent_orders') else 0
            my_total_orders = sum(context.get('orders_history', [])) if context.get('orders_history') else 0
            my_current_order = context.get('current_order', 0)
            my_pending_orders = context.get('pending_orders', [])
            my_total_in_transit = context.get('total_in_transit', 0)

            prompt += f"\n\n[CORE] My Total Order Status Analysis:\n"
            prompt += f"- My current period order: {my_current_order} units\n"
            prompt += f"- My last 3 periods total orders: {my_recent_orders} units\n"
            prompt += f"- My historical total orders: {my_total_orders} units\n"
            prompt += f"- My {'total WIP' if self.role == 'manufacturer' else 'total in-transit inventory'}: {my_total_in_transit} units\n"
            prompt += f"- My {'expected completion per period' if self.role == 'manufacturer' else 'pending shipment details'}: {my_pending_orders}\n"

            # 3. Supply-demand balance analysis
            prompt += f"\n[KEY] Supply-Demand Balance Analysis:\n"
            if retailer_demand_history and my_total_orders > 0:
                total_retailer_demand = sum(retailer_demand_history)
                balance_ratio = my_total_orders / max(total_retailer_demand, 1)
                prompt += f"- My orders / Retailer total demand ratio: {balance_ratio:.2f}\n"
                if balance_ratio > 1.5:
                    prompt += f"SEVERE WARNING: Your total orders significantly exceed retailer total demand by {balance_ratio:.1f}x - severe demand amplification!\n"
                    prompt += f"   Recommendation: Significantly reduce orders to align with actual retailer demand.\n"
                elif balance_ratio > 1.2:
                    prompt += f"WARNING: Your orders exceed retailer demand by {(balance_ratio-1)*100:.0f}%, consider moderate adjustment.\n"
                    prompt += f"   Recommendation: Reduce orders to better match actual retailer demand.\n"
                else:
                    prompt += f"GOOD: Your orders are basically balanced with retailer demand.\n"

            current_amplification = my_recent_orders / max(current_retail_demand, 1) if current_retail_demand > 0 else 0
            prompt += f"- Current order amplification factor: {current_amplification:.2f}x\n"
            if current_amplification > 2.0:
                prompt += f"SEVERE OVER-AMPLIFICATION: Your recent orders are {current_amplification:.1f}x retailer demand!\n"
            elif current_amplification > 1.5:
                prompt += f"SLIGHT AMPLIFICATION: Recommend controlling orders closer to retailer demand.\n"
            else:
                prompt += f"REASONABLE: Order quantity is relatively balanced with retailer demand.\n"

            prompt += f"\nBALANCE TARGET: Maintain balance between order quantity and actual retailer demand. Ensure all end-user demand is met!\n"

            # 6. Detailed upstream/downstream participant info sharing
            supply_chain_order = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
            current_index = supply_chain_order.index(self.role) if self.role in supply_chain_order else -1

            if current_index >= 0:
                prompt += f"\n[DETAILED] Upstream/Downstream Participant Status Analysis:\n"

                # Downstream info (closer to customer)
                if current_index > 0:
                    downstream_role = supply_chain_order[current_index - 1]
                    if downstream_role in shared:
                        downstream_info = shared[downstream_role]
                        downstream_orders_history = downstream_info.get('orders_history', [])
                        downstream_total_orders = sum(downstream_orders_history) if downstream_orders_history else 0

                        prompt += f"\nDOWNSTREAM ({downstream_role}):\n"
                        prompt += f"- Inventory status: {downstream_info.get('inventory', 'N/A')} units\n"
                        prompt += f"- In-transit inventory: {downstream_info.get('total_in_transit', 0)} units\n"
                        prompt += f"- Backorders: {downstream_info.get('backorder', 0)} units\n"
                        prompt += f"- Current demand: {downstream_info.get('demand', 'N/A')} units\n"
                        prompt += f"- Historical total orders: {downstream_total_orders} units\n"

                        # Downstream balance status
                        if downstream_role == 'retailer' and retailer_demand_history:
                            ds_balance = downstream_total_orders / max(sum(retailer_demand_history), 1)
                            prompt += f"- Downstream balance status: Order/Demand ratio {ds_balance:.2f} ({'BALANCED' if ds_balance <= 1.2 else 'IMBALANCED'})\n"

                # Upstream info (closer to production)
                if current_index < len(supply_chain_order) - 1:
                    upstream_role = supply_chain_order[current_index + 1]
                    if upstream_role in shared:
                        upstream_info = shared[upstream_role]
                        upstream_orders_history = upstream_info.get('orders_history', [])
                        upstream_total_orders = sum(upstream_orders_history) if upstream_orders_history else 0

                        prompt += f"\nUPSTREAM ({upstream_role}):\n"
                        prompt += f"- Inventory status: {upstream_info.get('inventory', 'N/A')} units\n"
                        prompt += f"- In-transit inventory: {upstream_info.get('total_in_transit', 0)} units\n"
                        prompt += f"- Received orders: {upstream_info.get('demand', 'N/A')} units\n"
                        prompt += f"- Historical total orders: {upstream_total_orders} units\n"

                        # Upstream production/shipment plan
                        if 'shipment_pipeline' in upstream_info:
                            pipeline_info = upstream_info.get('shipment_pipeline', [])
                            if pipeline_info:
                                prompt += f"- Production/Shipment plan: {pipeline_info} units (per-period plan)\n"
                                # Per-period breakdown (period 1 is nearest completion/arrival)
                                for idx, qty in enumerate(pipeline_info, start=1):
                                    if upstream_role == 'manufacturer':
                                        prompt += f"  . Period {idx} expected completion: {qty} units\n"
                                    else:
                                        prompt += f"  . Period {idx} expected arrival: {qty} units\n"
                                prompt += f"  . Note: These plans will be {'completed' if upstream_role == 'manufacturer' else 'delivered'} sequentially across the effective delivery lead time ({len(pipeline_info)} periods), which should correspond to your pipeline breakdown.\n"

                        # Upstream supply capacity assessment
                        if upstream_info.get('inventory', 0) + upstream_info.get('total_in_transit', 0) < my_recent_orders:
                            prompt += f"UPSTREAM SUPPLY TIGHT: May not fully meet your order demand\n"
                        else:
                            prompt += f"UPSTREAM SUPPLY ADEQUATE: Can meet your order demand\n"


            # 4. Full chain participant ordering analysis
            prompt += f"\n[IMPORTANT] Full Chain Participant Ordering Analysis:\n"
            total_chain_orders = 0
            total_chain_inventory = 0
            total_chain_in_transit = 0

            for agent_role, info in shared.items():
                if agent_role != self.role:
                    inventory_val = info.get('inventory', 0)
                    backorder = info.get('backorder', 0)
                    current_order = info.get('order', 0)
                    in_transit = info.get('total_in_transit', 0)
                    orders_history = info.get('orders_history', [])
                    total_orders = sum(orders_history) if orders_history else 0

                    total_chain_orders += total_orders
                    total_chain_inventory += inventory_val
                    total_chain_in_transit += in_transit

                    inventory_status = "Normal"
                    if backorder > 0:
                        inventory_status = "Backorder"
                    elif inventory_val > 20:
                        inventory_status = "Adequate"
                    elif inventory_val < 5:
                        inventory_status = "Tight"

                    prompt += f"- {agent_role}: current inventory {inventory_val} units ({inventory_status})\n"
                    prompt += f"  - Current order: {current_order} units, {'WIP' if agent_role == 'manufacturer' else 'in-transit'}: {in_transit} units\n"
                    prompt += f"  - Historical total orders: {total_orders} units, backorders: {backorder} units\n"

            # 5. Overall supply chain balance assessment
            if retailer_demand_history:
                total_retailer_demand = sum(retailer_demand_history)
                chain_demand_ratio = (total_chain_orders + my_total_orders) / max(total_retailer_demand, 1)

                prompt += f"\n[OVERALL ASSESSMENT] Supply Chain Balance Status:\n"
                prompt += f"- Retailer total demand: {total_retailer_demand} units\n"
                prompt += f"- Full chain total orders: {total_chain_orders + my_total_orders} units\n"
                # Removed imprecise full chain inventory display to avoid misleading supply-demand judgments
                prompt += f"- Full chain {'total WIP' if self.role == 'manufacturer' else 'total in-transit'}: {total_chain_in_transit + my_total_in_transit} units\n"

                chain_order_window_sum = 0
                lt_eval = context.get('lead_time', 0)
                for idx in range(lt_eval):
                    try:
                        orders_history = context.get('orders_history', [])
                        if orders_history:
                            chain_order_window_sum += (sum(orders_history[-lt_eval:]) if orders_history else 0)
                    except Exception:
                        pass
                my_orders_hist = context.get('orders_history', [])
                chain_order_window_sum += (sum(my_orders_hist[-lt_eval:]) if my_orders_hist else 0)

                # Total retail demand in LT window and per-period average (use current demand to fill if history is insufficient)
                demand_window_sum = (sum(retailer_demand_history[-lt_eval:]) if len(retailer_demand_history) >= lt_eval else current_retail_demand * lt_eval)
                avg_order_per_period = chain_order_window_sum / lt_eval
                avg_demand_per_period = demand_window_sum / lt_eval
                chain_avg_ratio = avg_order_per_period / max(avg_demand_per_period, 1)

                # Manufacturer uses lead-time-window-based supply-demand ratio (per-period average)
                if self.role == 'manufacturer':
                    my_pending = context.get('pending_orders', [])
                    my_inv = context.get('inventory', 0)
                    supply_window = my_inv + (sum(my_pending[:lt_eval]) if my_pending else 0)
                    supply_avg = supply_window / lt_eval
                    demand_avg = avg_demand_per_period
                    lt_avg_ratio = supply_avg / max(demand_avg, 1)
                    prompt += f"- Manufacturer LT period available supply (inventory + planned completion): {supply_window} units, average per period {supply_avg:.2f}\n"
                    prompt += f"- Supply-demand ratio considering lead time (per-period average): {lt_avg_ratio:.2f}\n"
                    if lt_avg_ratio > 2.0:
                        prompt += f"SEVERE IMBALANCE: Manufacturer average supply per period is {lt_avg_ratio:.1f}x end-user demand, systemic oversupply!\n"
                    elif lt_avg_ratio > 1.5:
                        prompt += f"SLIGHT IMBALANCE: Manufacturer average supply per period is elevated, recommend coordinating reduced output/ordering.\n"
                    else:
                        prompt += f"BASICALLY BALANCED: Manufacturer LT window average supply per period is reasonably matched with end-user demand.\n"
                    prompt += f"Note: Manufacturer must evaluate by lead time window, converting \"total supply\" and \"total demand\" within LT into per-period averages for comparison, to avoid total volume misjudgment.\n"
                else:
                    # Non-manufacturer roles use per-period average chain supply-demand ratio
                    prompt += f"- Overall supply-demand ratio (per-period average): {chain_avg_ratio:.2f}\n"
                    if chain_avg_ratio > 2.0:
                        prompt += f"SEVERE IMBALANCE: Full chain average orders per period are {chain_avg_ratio:.1f}x retailer demand, systemic over-ordering!\n"
                    elif chain_avg_ratio > 1.5:
                        prompt += f"SLIGHT IMBALANCE: Full chain average orders per period are elevated, recommend all roles coordinate to reduce orders.\n"
                    else:
                        prompt += f"BASICALLY BALANCED: Full chain average orders per period are reasonably aligned with retailer demand.\n"

            # 7. Key decision principle reminders
            prompt += f"\n\n[ULTIMATE GOAL] Supply-Demand Balance Decision Principles:\n"
            prompt += f"1. Ensure all end-user demand can be met\n"
            prompt += f"2. Keep your order quantity balanced with actual retailer demand\n"
            prompt += f"3. Avoid over-ordering that amplifies the bullwhip effect\n"
            prompt += f"4. Consider upstream and downstream inventory and in-transit situations\n"
            prompt += f"5. Maintain coordination and stability across the entire supply chain\n"

        # Add order quantity limit requirements
        agent_config = getattr(self.config, self.role)
        prompt += f"\n\n=== Order Limits ===\n"

        # Check if discrete point ordering is enabled
        if (hasattr(agent_config, 'enable_discrete_points') and
            agent_config.enable_discrete_points and
            hasattr(agent_config, 'discrete_order_points') and
            agent_config.discrete_order_points):

            discrete_points = agent_config.discrete_order_points
            # Filter out 0 or negative discrete points
            valid_discrete_points = [p for p in discrete_points if p > 0]
            if not valid_discrete_points:
                valid_discrete_points = [4, 8, 12, 16, 20]  # default values

            discrete_points_str = ", ".join(map(str, valid_discrete_points))
            prompt += f"\n=== Discrete Point Ordering Mode ===\n"
            prompt += f"- Available discrete points: {discrete_points_str} units\n"
            prompt += f"- Strict constraint: Your order quantity MUST be selected from one of the above discrete points!\n"
            prompt += f"- Prohibited: Do NOT select any value outside the discrete points, including 0 or any other value!\n"
            prompt += f"- Selection requirement: Choose the most appropriate value from [{discrete_points_str}] based on supply-demand conditions\n\n"

        # Continuous range limits
        elif hasattr(agent_config, 'min_order_quantity') and hasattr(agent_config, 'max_order_quantity'):
            min_order = agent_config.min_order_quantity
            max_order = agent_config.max_order_quantity
            prompt += f"\n=== Order Limit Constraints ===\n"
            prompt += f"- Minimum order quantity: {min_order} units\n"
            prompt += f"- Maximum order quantity: {max_order} units\n"
            prompt += f"- Mandatory requirement: Your order MUST be strictly within the {min_order}-{max_order} range!\n"
            prompt += f"- Consequence: Orders outside the range will be automatically adjusted by the system, which may affect decision effectiveness\n\n"

        #prompt += "\n\n=== Decision Requirements ===\nBased on the above complete information, determine your order quantity.\n\nImportant: Only respond with a positive integer, do not include any text, explanation, or other content.\n\nExample output: 12"

        return prompt

    def set_coordinator_guidance(self, guidance: str):
        """Set coordinator guidance"""
        self.coordinator_guidance = guidance
        self.logger.info(f"Received coordinator guidance: {guidance[:50]}...")

    def get_coordinator_guidance(self) -> str:
        """Get coordinator guidance"""
        return self.coordinator_guidance if self.coordinator_guidance is not None else ""

    def get_last_reasoning(self) -> str:
        """Get the reasoning process of the last decision"""
        return self.last_reasoning

    def get_decision_details(self) -> dict:
        """Get decision details including reasoning process and decision result"""
        return {
            'reasoning': self.last_reasoning,
            'coordinator_guidance': self.coordinator_guidance,
            'role': self.role
        }



    def _parse_llm_response(self, response: str) -> int:
        """Parse LLM response to extract order quantity and reasoning - enhanced JSON format parsing"""
        try:
            print(f"\n--- [{self.role}] Start parsing LLM response ---")
            print(f"Raw response: '{response}'")

            # Clean response
            if response is None:
                print(f"ERROR: [{self.role}] Response is None, using fallback")
                return self._fallback_decision()

            response = response.strip()
            print(f"Cleaned response: '{response}'")
            print(f"Length after cleaning: {len(response)}")

            # Check if empty
            if not response:
                print(f"ERROR: [{self.role}] Response is empty, using fallback")
                return self._fallback_decision()

            # Save complete response as reasoning process (for later analysis)
            self.last_reasoning = response

            # Attempt to extract JSON portion from response (handling cases with reasoning content)
            cleaned_response = self._extract_json_from_response(response)
            print(f"Extracted JSON portion: '{cleaned_response}'")

            # Prioritize JSON format parsing
            order_quantity, reason = self._parse_json_decision_with_reason(cleaned_response)

            # If JSON parsing succeeded, save the reason explanation
            if order_quantity is not None and reason is not None:
                self.last_decision_reason = reason
                print(f"DEBUG: [{self.role}] JSON parsing succeeded - order quantity: {order_quantity}, reason: {reason}")
            else:
                # Fallback: attempt number extraction (compatible with old format)
                order_quantity = self._extract_number_from_response(response)
                self.last_decision_reason = "No reason provided"
                print(f"DEBUG: [{self.role}] Using number extraction fallback: {order_quantity}")

            print(f"DEBUG: [{self.role}] Final extracted number: {order_quantity}")
            print(f"--- [{self.role}] Parsing ended ---\n")

            return self._apply_order_limits(order_quantity)

        except Exception as e:
            self.logger.error(f"Error parsing LLM response '{response}': {e}")
            print(f"DEBUG: [{self.role}] Parsing exception, using fallback: {e}")
            self.last_decision_reason = "Parsing exception, using default decision"
            return self._fallback_decision()

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON portion from response, handling cases with reasoning content"""
        import re

        # Method 1: Find complete JSON object {key: value, key: value}
        json_pattern = r'\{[^{}]*"订货决策"[^{}]*"原因解释"[^{}]*\}'
        json_matches = re.findall(json_pattern, response)
        if json_matches:
            print(f"DEBUG: Found complete JSON object: {json_matches[0]}")
            return json_matches[0]

        # Method 2: Find any JSON object
        json_pattern2 = r'\{[^{}]*\}'
        json_matches2 = re.findall(json_pattern2, response)
        if json_matches2:
            # Prioritize JSON containing relevant fields
            for match in json_matches2:
                if '订货决策' in match or 'decision' in match or 'order_decision' in match:
                    print(f"DEBUG: Found JSON object: {match}")
                    return match
            # If no relevant field found, return first JSON object
            print(f"DEBUG: Using first JSON object: {json_matches2[0]}")
            return json_matches2[0]

        # Method 3: No JSON found, return raw response
        print(f"DEBUG: No JSON object found, returning raw response")
        return response

    def _extract_number_from_response(self, response: str) -> int:
        """Enhanced logic to extract numbers from LLM response"""
        import re

        print(f"DEBUG: [{self.role}] Starting number extraction...")

        # Method 1: Check if pure numeric (ideal case)
        print(f"Check if pure numeric: {response.isdigit()}")
        if response.isdigit():
            result = int(response)
            print(f"Method 1 success: pure number {result}")
            return result

        # Method 2: Find standalone numbers (surrounded by spaces, punctuation, or line boundaries)
        standalone_numbers = re.findall(r'(?:^|\s|[^\d])(\d+)(?:\s|[^\d]|$)', response)
        print(f"Method 2: standalone numbers {standalone_numbers}")
        if standalone_numbers:
            # Filter numbers in reasonable range, prioritize reasonable order quantities
            valid_numbers = [int(n) for n in standalone_numbers if 0 <= int(n) <= 1000]
            if valid_numbers:
                # Smart selection: prioritize numbers in 3-100 range (more likely to be order quantities)
                preferred_numbers = [n for n in valid_numbers if 3 <= n <= 100]
                if preferred_numbers:
                    result = preferred_numbers[-1]  # Take last preferred number
                    print(f"Method 2 success: preferred number {result}")
                    return result
                else:
                    result = valid_numbers[-1]  # Take last reasonable number
                    print(f"Method 2 success: standalone number {result}")
                    return result

        # Method 3: Find numbers at end of line (usually the final decision)
        line_end_numbers = re.findall(r'(\d+)\s*$', response, re.MULTILINE)
        print(f"Method 3: end of line numbers {line_end_numbers}")
        if line_end_numbers:
            valid_numbers = [int(n) for n in line_end_numbers if 0 <= int(n) <= 1000]
            if valid_numbers:
                result = valid_numbers[-1]
                print(f"Method 3 success: end of line number {result}")
                return result

        # Method 4: Find numbers at end of sentences
        sentence_end_numbers = re.findall(r'(\d+)[.!?]*\s*$', response)
        print(f"Method 4: end of sentence numbers {sentence_end_numbers}")
        if sentence_end_numbers:
            valid_numbers = [int(n) for n in sentence_end_numbers if 0 <= int(n) <= 1000]
            if valid_numbers:
                result = valid_numbers[-1]
                print(f"Method 4 success: end of sentence number {result}")
                return result

        # Method 5: Find all numbers, smart filtering
        all_numbers = re.findall(r'\d+', response)
        print(f"Method 5: all numbers {all_numbers}")
        if all_numbers:
            # Convert to integers and filter
            int_numbers = [int(n) for n in all_numbers]

            # Filter out numbers that are clearly not order quantities (e.g. costs, percentages, rounds, etc.)
            filtered_numbers = []
            for num in int_numbers:
                # Keep reasonable order quantity range
                if 0 <= num <= 1000:
                    # Exclude obvious cost numbers (numbers after decimal points are usually costs)
                    if num not in [0, 1, 2] or len([x for x in int_numbers if x == num]) == 1:
                        filtered_numbers.append(num)

            if filtered_numbers:
                result = filtered_numbers[-1]  # Take last reasonable number
                print(f"Method 5 success: filtered number {result}")
                return result
            elif int_numbers:
                # If no numbers after filtering, take the last raw number
                result = int_numbers[-1]
                print(f"Method 5 fallback: last number {result}")
                return min(max(result, 0), 1000)  # Constrain within reasonable range

        # Method 6: Special handling for incomplete responses (e.g. "We need")
        incomplete_patterns = [
            r'we need.*?(\d+)',
            r'i recommend.*?(\d+)',
            r'order.*?(\d+)',
            r'quantity.*?(\d+)',
            r'decision.*?(\d+)'
        ]

        for pattern in incomplete_patterns:
            match = re.search(pattern, response.lower())
            if match:
                result = int(match.group(1))
                if 0 <= result <= 1000:
                    print(f"Method 6 success: incomplete response match {result}")
                    return result

        # Method 7: If response is incomplete text, estimate a reasonable value
        if 'we need' in response.lower() or 'i recommend' in response.lower():
            print(f"DEBUG: [{self.role}] Detected incomplete response, using intelligent fallback")
            return self._intelligent_fallback_for_incomplete_response(response)

        # Method 8: If no numbers at all, return default value
        print(f"WARNING: [{self.role}] Cannot extract any reasonable number from response, using default 0")
        print(f"WARNING: [{self.role}] Raw response: {response}")
        return 0

    def _intelligent_fallback_for_incomplete_response(self, response: str) -> int:
        """Intelligent fallback strategy for incomplete responses"""
        print(f"DEBUG: [{self.role}] Executing intelligent fallback for incomplete response")

        # If response starts with "We need", this may mean LLM wanted to say "We need X units"
        # Estimate a reasonable value based on context
        try:
            context = self.get_decision_context()
            inventory = context.get('inventory', 0)
            backorder = context.get('backorder', 0)
            demand = context.get('current_demand', 0)

            # Simple heuristic: if LLM says "We need", it likely needs to replenish inventory
            if backorder > 0:
                estimated_order = backorder + demand  # replenish backorders plus demand
            elif inventory < demand:
                estimated_order = demand - inventory + 2  # replenish to meet demand plus small buffer
            else:
                estimated_order = max(demand, 5)  # Conservative ordering under normal conditions

            # Constrain within reasonable range
            estimated_order = max(1, min(estimated_order, 50))

            print(f"DEBUG: [{self.role}] Estimated order based on context: {estimated_order} (inventory={inventory}, backorder={backorder}, demand={demand})")
            return estimated_order

        except Exception as e:
            print(f"DEBUG: [{self.role}] Intelligent fallback calculation failed: {e}, using default")
            return 5  # Default conservative value

    def _apply_order_limits(self, order_quantity: int) -> int:
        """Apply order quantity limits"""
        agent_config = getattr(self.config, self.role)
        original_quantity = order_quantity

        # Discrete point selection logic
        if (hasattr(agent_config, 'enable_discrete_points') and
            agent_config.enable_discrete_points and
            hasattr(agent_config, 'discrete_order_points') and
            agent_config.discrete_order_points):

            discrete_points = agent_config.discrete_order_points

            # Ensure discrete point list does not contain 0 or negative numbers
            valid_discrete_points = [p for p in discrete_points if p > 0]

            if not valid_discrete_points:
                # If no valid discrete points, use defaults
                valid_discrete_points = [4, 8, 12, 16, 20]
                print(f"WARNING: No valid discrete points (>0) found, using default: {valid_discrete_points}")

            # If original order quantity <= 0, select smallest valid discrete point
            if original_quantity <= 0:
                order_quantity = min(valid_discrete_points)
                print(f"DEBUG: Original quantity <= 0, using minimum discrete point: {order_quantity}")
            else:
                # Find closest discrete point
                order_quantity = min(valid_discrete_points, key=lambda x: abs(x - original_quantity))

            print(f"DEBUG: Applied discrete points selection: original={original_quantity}, adjusted={order_quantity}, points={valid_discrete_points}")

            # Update decision explanation to reflect adjustment
            if hasattr(self, 'last_decision_explanation'):
                self.last_decision_explanation = getattr(self, 'last_decision_explanation', '') + f" (adjusted: {original_quantity}->discrete {order_quantity})"

            return order_quantity

        # Continuous range limit logic
        # Add debug log
        print(f"DEBUG: Checking order limits for {self.role}: min={getattr(agent_config, 'min_order_quantity', None)}, max={getattr(agent_config, 'max_order_quantity', None)}")

        if (hasattr(agent_config, 'min_order_quantity') and hasattr(agent_config, 'max_order_quantity') and
            agent_config.min_order_quantity is not None and agent_config.max_order_quantity is not None):
            min_order = agent_config.min_order_quantity
            max_order = agent_config.max_order_quantity

            order_quantity = max(min_order, min(max_order, order_quantity))

            print(f"DEBUG: Applying order limits: original={original_quantity}, adjusted={order_quantity}, limits=[{min_order}, {max_order}]")

            if original_quantity != order_quantity:
                print(f"DEBUG: Order quantity adjusted from {original_quantity} to {order_quantity} due to limits [{min_order}, {max_order}]")
                # Update decision explanation to reflect adjustment
                if hasattr(self, 'last_decision_explanation'):
                    self.last_decision_explanation = getattr(self, 'last_decision_explanation', '') + f" (adjusted: {original_quantity}->clamped {order_quantity})"

        return order_quantity

    def _is_harmony_format(self, response: str) -> bool:
        """Detect if harmony format response"""
        # harmony format markers <mcreference link="https://cobusgreyling.medium.com/what-is-gpt-oss-harmony-response-format-a29f266d6672" index="3">3</mcreference>
        harmony_markers = ['<|start|>', '<|end|>', '<|message|>', '<|channel|>', '<|call|>']
        return any(marker in response for marker in harmony_markers)

    def _parse_harmony_response(self, response: str) -> int:
        """Parse harmony format response with enhanced fault tolerance <mcreference link="https://cookbook.openai.com/articles/openai-harmony" index="2">2</mcreference>"""
        import re
        import json

        try:
            # Extract text from main content channel
            # Harmony format may contain multiple channels; find the one with the decision

            # Method 1: Find content between <|message|> markers
            message_pattern = r'<\|message\|>(.*?)(?:<\|end\||$)'
            message_matches = re.findall(message_pattern, response, re.DOTALL)

            # Method 2: Find plain text without special markers
            # Remove all harmony control markers to get plain text
            clean_text = re.sub(r'<\|[^|]+\|>', '', response)

            # Attempt to extract decision from message content
            content_to_parse = ""
            if message_matches:
                content_to_parse = message_matches[-1].strip()  # Take last message
            else:
                content_to_parse = clean_text.strip()

            # Use multiple regex patterns to extract order quantity
            patterns = [
                r'order quantity[：:]*(\d+)',  # English format
                r'order[：:]*(\d+)',           # Short English format
                r'quantity[：:]*(\d+)',        # Quantity format
                r'decision[：:]*(\d+)',        # Decision format
                r'订货决策[：:]*(\d+)',         # Chinese format
                r'订购数量[：:]*(\d+)',         # Standard Chinese format
                r'订购[：:]*(\d+)',             # Simplified Chinese format
                r'数量[：:]*(\d+)',             # Quantity format
                r'决策[：:]*(\d+)',             # Decision format
            ]

            order_quantity = None
            for pattern in patterns:
                match = re.search(pattern, content_to_parse, re.IGNORECASE)
                if match:
                    order_quantity = int(match.group(1))
                    break

            # If no specific format found, try traditional delimiter method
            if order_quantity is None:
                if '|' in content_to_parse:
                    parts = content_to_parse.split('|', 1)
                    quantity_str = parts[0].strip()
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                else:
                    # Attempt to extract number and explanation
                    quantity_str = content_to_parse
                    explanation = content_to_parse

                # Extract number
                numbers = re.findall(r'\d+', quantity_str)
                if numbers:
                    order_quantity = int(numbers[0])

            if order_quantity is not None:
                # Extract explanation portion (multiple patterns)
                explanation_patterns = [
                    r'\((.*?)\)',           # Parenthesized content
                    r'解释[：:]*(.*?)(?:\n|$)',  # Content after explanation
                    r'原因[：:]*(.*?)(?:\n|$)',  # Content after reason
                ]

                explanation = "Harmony format parsing"
                for exp_pattern in explanation_patterns:
                    exp_match = re.search(exp_pattern, content_to_parse, re.IGNORECASE)
                    if exp_match:
                        explanation = exp_match.group(1).strip()
                        break

                # Apply order quantity limits
                order_quantity = self._apply_order_limits(order_quantity)

                # Validate final order quantity
                agent_config = getattr(self.config, self.role)
                if (hasattr(agent_config, 'enable_discrete_points') and
                    agent_config.enable_discrete_points):
                    # In discrete point mode, order quantity must be > 0
                    if order_quantity > 0 and order_quantity <= 1000:
                        # Save decision explanation
                        self.last_decision_explanation = f"Harmony format parsing: {explanation}"
                        self.logger.info(f"Successfully parsed harmony format response: order {order_quantity}, explanation: {explanation[:50]}...")
                        return order_quantity
                else:
                    # In continuous mode, order quantity must be non-negative
                    if 0 <= order_quantity <= 1000:
                        # Save decision explanation
                        self.last_decision_explanation = f"Harmony format parsing: {explanation}"
                        self.logger.info(f"Successfully parsed harmony format response: order {order_quantity}, explanation: {explanation[:50]}...")
                        return order_quantity

            # If unable to extract valid number from harmony format, log details
            self.logger.warning(f"Harmony format parsing failed, response content: {response[:200]}...")

        except Exception as e:
            self.logger.warning(f"Harmony format parsing exception: {e}, response: {response[:100]}...")

        # Parse failed, return default value for upper layer to handle
        return 0

    def _parse_json_decision_with_reason(self, response: str) -> tuple:
        """Parse JSON format decision response, return order quantity and reason"""
        import json
        try:
            # Attempt direct JSON parsing
            data = json.loads(response)
            if isinstance(data, dict):
                # Find order decision field
                order_quantity = None
                reason = None

                # Lookup Chinese fields first (backward compatible)
                if '订货决策' in data:
                    order_quantity = data['订货决策']
                if '原因解释' in data:
                    reason = data['原因解释']

                # Lookup English fields (new format)
                if order_quantity is None and 'order_decision' in data:
                    order_quantity = data['order_decision']
                if reason is None and 'reasoning' in data:
                    reason = data['reasoning']

                # Fallback: lookup English fields
                if order_quantity is None:
                    for key in ['decision', 'order_quantity', 'quantity', 'order']:
                        if key in data and isinstance(data[key], (int, float)):
                            order_quantity = int(data[key])
                            break

                if reason is None:
                    for key in ['reason', 'explanation', 'reasoning']:
                        if key in data and isinstance(data[key], str):
                            reason = data[key]
                            break

                # Validate data validity
                if order_quantity is not None and isinstance(order_quantity, (int, float)):
                    order_quantity = int(order_quantity)
                    # Check if discrete point mode is enabled
                    agent_config = getattr(self.config, self.role)
                    if (hasattr(agent_config, 'enable_discrete_points') and
                        agent_config.enable_discrete_points):
                        # In discrete point mode, order quantity must be > 0
                        if order_quantity > 0:
                            reason = reason or "No reason provided"
                            return order_quantity, reason
                        else:
                            print(f"DEBUG: [{self.role}] Rejecting zero or negative order quantity in discrete point mode: {order_quantity}")
                    else:
                        # In continuous mode, order quantity must be non-negative
                        if order_quantity >= 0:
                            reason = reason or "No reason provided"
                            return order_quantity, reason

        except json.JSONDecodeError as e:
            print(f"DEBUG: [{self.role}] JSON parsing failed: {e}")
        except Exception as e:
            print(f"DEBUG: [{self.role}] JSON parsing exception: {e}")

        # Parse failed, return None
        return None, None

    def _parse_json_decision(self, response: str) -> int:
        """Parse JSON format decision response (maintain compatibility)"""
        import json
        order_quantity, _ = self._parse_json_decision_with_reason(response)
        if order_quantity is not None:
            return order_quantity

        # Fallback: old version parsing logic
        try:
            # Attempt direct JSON parsing
            data = json.loads(response)
            if isinstance(data, dict):
                # Find possible order quantity fields
                for key in ['decision', 'order_quantity', 'quantity', 'order']:
                    if key in data and isinstance(data[key], (int, float)):
                        self.last_decision_explanation = data.get('explanation', 'JSON format response')
                        return int(data[key])
            elif isinstance(data, (int, float)):
                self.last_decision_explanation = 'JSON numeric response'
                return int(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return 0

    def _fallback_decision(self) -> int:
        """Fallback decision logic (when LLM fails), intelligent decision based on current state"""
        try:
            # Get current state
            current_inventory = getattr(self.state, 'inventory', 0)
            current_backorder = getattr(self.state, 'backorder', 0)
            current_demand = getattr(self, 'current_demand', 0)

            # Calculate in-transit inventory
            pipeline_total = sum(getattr(self, 'shipment_pipeline', []))

            # Intelligent fallback strategy: based on demand and inventory status
            if current_backorder > 0:
                # Has backorders, prioritize replenishing backorders
                needed = current_backorder + current_demand
            elif current_inventory < current_demand:
                # Insufficient inventory, replenish to safety level
                safety_stock = max(current_demand * 1.5, 5)
                needed = safety_stock - current_inventory + current_demand
            else:
                # Normal situation, maintain basic demand
                needed = current_demand

            # Consider in-transit inventory
            final_order = max(0, int(needed - pipeline_total))

            # Apply order limits
            final_order = self._apply_order_limits(final_order)

            self.logger.info(f"Using intelligent fallback decision: {final_order} (demand={current_demand}, inventory={current_inventory}, backorder={current_backorder})")
            return final_order

        except Exception as e:
            self.logger.error(f"Fallback decision error: {e}")
            # Final safety net strategy
            return self._apply_order_limits(max(getattr(self, 'current_demand', 0), 5))

    def make_decision(self) -> int:
        """Make ordering decision using LLM with integrated overreaction suppression mechanism"""
        if not self.llm_client:
            self.logger.warning("No LLM client available, using fallback decision")
            return self._fallback_decision()

        try:
            # Get decision context
            context = self.get_decision_context()

            # Record demand signal history
            current_demand = context.get('current_demand', 0)
            self.demand_signal_history.append(current_demand)

            # Add coordinator guidance to context
            if self.coordinator_guidance:
                context['coordinator_guidance'] = self.coordinator_guidance


            # Create prompts
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(context)

            # Debug log: LLM input (detailed)
            print(f"\n=== [{self.role}] LLM call started ===")
            print(f"System prompt (complete): {system_prompt}")
            print(f"User prompt: {user_prompt}")
            print(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")

            # Call LLM
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Debug log: LLM output (detailed)
            print(f"LLM raw response: '{response}'")
            print(f"Response length: {len(response) if response else 0}")
            print(f"Response type: {type(response)}")
            print(f"=== [{self.role}] LLM call ended ===")

            # Parse response
            final_order_quantity = self._parse_llm_response(response)

            # No need for reasoning explanation, just record decision
            self.last_reasoning = f"Decision: {final_order_quantity}"

            # Record decision history (including explanation and reason)
            decision_record = {
                'round': context.get('current_round', 0),
                'demand': current_demand,
                'inventory': context.get('inventory', 0),
                'backorder': context.get('backorder', 0),
                'decision': final_order_quantity,
                'explanation': getattr(self, 'last_decision_explanation', 'No explanation provided'),
                'reason': getattr(self, 'last_decision_reason', 'No reason provided')
            }
            self.decision_history.append(decision_record)
            self.decision_reasons_history.append(decision_record)

            self.logger.debug(f"LLM decision: {final_order_quantity} (response: {response.strip()})")
            return final_order_quantity

        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")
            return self._fallback_decision()

    def get_decision_explanation(self) -> Dict[str, Any]:
        """Get decision explanation (for analysis)"""
        context = self.get_decision_context()

        explanation = {
            'agent_role': self.role,
            'round': self.current_round,
            'decision_context': context,
            'system_prompt': self._create_system_prompt(),
            'user_prompt': self._create_user_prompt(context)
        }

        return explanation




class RuleBasedAgent(BaseAgent):
    """Rule-based agent (for comparison)"""

    def __init__(self,
                 role: str,
                 config: GameConfig,
                 strategy: str = 'base_stock',
                 agent_id: str = ""):
        """
        Initialize rule-based agent

        Args:
            role: Agent role
            config: Game configuration
            strategy: Strategy type ('base_stock', 'order_up_to', 'moving_average')
            agent_id: Agent ID
        """
        super().__init__(role, config, None, agent_id)
        self.strategy = strategy

        # Strategy parameters
        self.base_stock_level = 20  # Base stock level
        self.order_up_to_level = 25  # Order-up-to point
        self.ma_window = 5  # Moving average window

    def make_decision(self) -> int:
        """Make decision based on rules"""
        if self.strategy == 'base_stock':
            order_quantity = self._base_stock_policy()
        elif self.strategy == 'order_up_to':
            order_quantity = self._order_up_to_policy()
        elif self.strategy == 'moving_average':
            order_quantity = self._moving_average_policy()
        else:
            order_quantity = self._simple_policy()

        # Apply order limits
        return self._apply_order_limits(order_quantity)

    def _base_stock_policy(self) -> int:
        """Base stock policy"""
        current_position = self.state.inventory + sum(self.shipment_pipeline)
        target = self.base_stock_level
        order = max(0, target - current_position + self.current_demand)
        return order

    def _order_up_to_policy(self) -> int:
        """Order-up-to policy"""
        current_position = self.state.inventory + sum(self.shipment_pipeline)
        if current_position <= self.order_up_to_level:
            return self.current_demand * 2  # Order twice the demand
        return self.current_demand

    def _moving_average_policy(self) -> int:
        """Moving average policy"""
        if len(self.state.orders_history) < self.ma_window:
            return self.current_demand

        # Calculate moving average of recent demands
        recent_demands = [self.current_demand] + self.state.orders_history[-self.ma_window+1:]
        avg_demand = sum(recent_demands) / len(recent_demands)

        # Decision based on average demand and current inventory
        target_inventory = avg_demand * (self.lead_time + 1)
        current_position = self.state.inventory + sum(self.shipment_pipeline)
        order = max(0, target_inventory - current_position + self.state.backorder)

        return int(order)

    def _simple_policy(self) -> int:
        """Simple policy: order equals demand"""
        return self.current_demand

    def _apply_order_limits(self, order_quantity: int) -> int:
        """Apply order quantity limits"""
        agent_config = getattr(self.config, self.role)
        original_quantity = order_quantity

        # Discrete point selection logic
        if (hasattr(agent_config, 'enable_discrete_points') and
            agent_config.enable_discrete_points and
            hasattr(agent_config, 'discrete_order_points') and
            agent_config.discrete_order_points):

            discrete_points = agent_config.discrete_order_points

            # Ensure discrete point list does not contain 0 or negative numbers
            valid_discrete_points = [p for p in discrete_points if p > 0]

            if not valid_discrete_points:
                # If no valid discrete points, use defaults
                valid_discrete_points = [4, 8, 12, 16, 20]
                print(f"WARNING: No valid discrete points (>0) found, using default: {valid_discrete_points}")

            # If original order quantity <= 0, select smallest valid discrete point
            if original_quantity <= 0:
                order_quantity = min(valid_discrete_points)
                print(f"DEBUG: Original quantity <= 0, using minimum discrete point: {order_quantity}")
            else:
                # Find closest discrete point
                order_quantity = min(valid_discrete_points, key=lambda x: abs(x - original_quantity))

            print(f"DEBUG: Applied discrete points selection: original={original_quantity}, adjusted={order_quantity}, points={valid_discrete_points}")

            # Update decision explanation to reflect adjustment
            if hasattr(self, 'last_decision_explanation'):
                self.last_decision_explanation = getattr(self, 'last_decision_explanation', '') + f" (adjusted: {original_quantity}->discrete {order_quantity})"

            return order_quantity

        # Continuous range limit logic
        # Add debug log
        print(f"DEBUG: Checking order limits for {self.role}: min={getattr(agent_config, 'min_order_quantity', None)}, max={getattr(agent_config, 'max_order_quantity', None)}")

        if (hasattr(agent_config, 'min_order_quantity') and hasattr(agent_config, 'max_order_quantity') and
            agent_config.min_order_quantity is not None and agent_config.max_order_quantity is not None):
            min_order = agent_config.min_order_quantity
            max_order = agent_config.max_order_quantity

            order_quantity = max(min_order, min(max_order, order_quantity))

            print(f"DEBUG: Applying order limits: original={original_quantity}, adjusted={order_quantity}, limits=[{min_order}, {max_order}]")

            if original_quantity != order_quantity:
                print(f"DEBUG: Order quantity adjusted from {original_quantity} to {order_quantity} due to limits [{min_order}, {max_order}]")
                # Update decision explanation to reflect adjustment
                if hasattr(self, 'last_decision_explanation'):
                    self.last_decision_explanation = getattr(self, 'last_decision_explanation', '') + f" (adjusted: {original_quantity}->clamped {order_quantity})"

        return order_quantity
