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

        # Decision history
        self.decision_history = []
        self.demand_signal_history = []

        # Store last reasoning process
        self.last_reasoning = ""
        self.last_raw_response = ""

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
        holding = getattr(self.config, self.role).cost_config.holding_cost
        backorder = getattr(self.config, self.role).cost_config.backorder_cost
        ratio = backorder / holding
        is_thinking = getattr(self.config.llm, 'is_thinking_model', False)

        # Role-specific strategic guidance
        role_guidance = {
            'retailer': """=== Your Role: Retailer ===
You face end-customer demand directly. Your advantage is seeing real market signals without upstream amplification.
- Focus on demand forecasting accuracy — use historical demand patterns to anticipate future needs
- Maintain adequate safety stock because you directly bear backorder costs from customer-facing shortages
- Avoid over-ordering when demand spikes temporarily — distinguish signal from noise
- Your orders directly influence the entire supply chain upstream — stabilize your ordering pattern""",
            'wholesaler': """=== Your Role: Wholesaler ===
You sit between retailer and distributor. Your demand is the retailer's orders, which may be amplified.
- Be aware that retailer orders may already contain bullwhip amplification — do not amplify further
- Monitor whether retailer orders exceed end-customer demand (if information sharing is enabled)
- Balance your inventory against the retailer's ability to absorb shipments
- Consider the distributor's supply reliability when determining your safety stock level""",
            'distributor': """=== Your Role: Distributor ===
You are furthest from the end customer. Your demand signals pass through two stages of potential amplification.
- Your demand (wholesaler orders) may be significantly amplified — apply dampening to your orders
- Pay close attention to total chain inventory levels to gauge overall supply-demand balance
- Consider both upstream (manufacturer production capacity) and downstream (wholesaler demand) constraints
- Prioritize supply chain stability over aggressive cost minimization""",
            'manufacturer': """=== Your Role: Manufacturer ===
You produce goods with a production lead time. You do not simply order — you plan production.
- Production takes {plt} periods — you must forecast demand {plt} periods ahead
- Raw materials and production capacity are assumed unlimited, but production time is fixed
- Use the production pipeline (WIP) to smooth output — avoid frequent large changes in production quantity
- Your production decisions determine the entire supply chain's supply capability — prioritize stability
- Consider the full pipeline: inventory + WIP covers demand over the production lead time horizon""".format(plt=self.production_lead_time),
        }

        return f"""You are a {self.role} in a supply chain making ordering decisions.

=== Identity and Responsibilities ===
You are the {self.role} in the supply chain, responsible for making ordering decisions to optimize overall supply chain performance.
Your core responsibility is to minimize total costs while meeting downstream demand.

=== Core Objectives ===
Primary goal: Cost minimization
- Total cost = Holding cost + Backorder cost
- Holding cost: {holding} yuan per unit per period
- Backorder cost: {backorder} yuan per unit per period
- Key tradeoff: Backorder cost is {ratio:.1f}x holding cost, balance carefully

=== Strategic Framework: Dual-Horizon Supply-Demand Balance ===
Make decisions by evaluating TWO time horizons:

SHORT-TERM (current period):
- Net position = inventory + incoming_shipment_this_period - backorders - current_demand
- Can you meet immediate demand? A negative net position means stockout this period.
- Short-term shortage is more costly (backorder_cost = {ratio:.1f}x holding_cost)

LONG-TERM (over effective lead time = {self.lead_time} periods):
- Total supply = inventory + all in-transit/WIP arriving within lead time
- Total expected demand = average_demand × lead_time periods
- Coverage ratio = total_supply / total_expected_demand
- Ratio > 1.5: oversupply risk (bullwhip amplification) — consider reducing orders
- Ratio < 1.0: undersupply risk — need to increase orders to avoid persistent backorders
- Target ratio: 1.1-1.3 (slight buffer to handle demand variability)

{role_guidance.get(self.role, '')}

=== Supply Chain Environment ===
Operating environment:
- Lead time: Effective delivery time is {self.lead_time} periods ({('Order '+str(self.order_lead_time)+'+Transport '+str(self.transport_lead_time)) if self.role != 'manufacturer' else ('Production '+str(self.production_lead_time))})
{'' if self.role == 'manufacturer' else f'- Upstream shipment info: The system will provide next-period arrival quantities, detailed pending shipments for each period, and the last {2 * self.lead_time} periods of shipment history. Consider these in your decision.'}
- Information flow: Demand information propagates upstream in stages, with inherent delays
{'- Order constraints: No restrictions — you may choose any non-negative integer order quantity freely.' if getattr(self.config, self.role).no_decision_constraints else '- Order constraints: Must be non-negative integers (>=0); whether zero orders are allowed depends on system settings'}
{'' if getattr(self.config, self.role).no_decision_constraints else '- Order limits: If the system has set order limit conditions (min/max quantity or discrete point constraints), you must strictly follow them'}
- Processing rule: Orders are processed on a First-In-First-Out (FIFO) basis

=== Decision Task ===
You need to make an ordering decision based on current state information:
1. Assess short-term position: Can you meet current demand? What is the immediate net inventory?
2. Assess long-term balance: Over the lead time horizon, is supply adequate relative to expected demand?
3. Evaluate historical demand trends and cost patterns
4. Weigh holding costs against backorder risk, accounting for the {ratio:.1f}x cost ratio
5. Determine the optimal order quantity to minimize total cost while maintaining supply-demand balance

=== Output Format Requirements ===
CRITICAL REQUIREMENT - OUTPUT FORMAT:
You must strictly follow these requirements:

1. Output ONLY a single JSON object as your final response — no extra text before or after
2. Use exact format: {{"decision": <number>, "reason": "<text>"}}
3. "decision" must be a non-negative integer (0 or greater)
4. "reason" must explain your decision concisely (50-150 characters), addressing:
   - Current net inventory position (short-term)
   - Supply-demand balance over lead time (long-term)
   - Why this order quantity is appropriate

Correct output examples:
{{"decision": 5, "reason": "ST: net +3, current period OK. LT: coverage 0.8 (<1.0), pipeline too thin — need to rebuild pipeline. Order slightly above demand to restore balance."}}
{{"decision": 0, "reason": "ST: net +12, ample inventory. LT: coverage 1.5 (>1.5), oversupplied — skip order to let pipeline drain and reduce holding costs."}}
{{"decision": 8, "reason": "ST: net -2, immediate shortage — must cover backorder now. LT: coverage 0.6 (<1.0), urgent replenishment needed. Order covers demand plus backorder recovery."}}

Incorrect output examples (PROHIBITED):
- Plain number: 5
- With explanation: I suggest ordering 5 units, because...
- Wrong format: {{"order_decision": 5}}
- With reasoning text: Based on analysis, {{"decision": 5, "reason": "need restock"}}

Important reminders:
- {'Your internal thinking is handled by the system — put your decision reasoning ONLY into the JSON reasoning field. Output the JSON directly.' if is_thinking else 'Do NOT output any reasoning process or thinking steps — output the JSON object directly, nothing else.'}
- Ensure the JSON format is complete and parseable
- Your reasoning field should reference specific numbers from the data provided
        """

    def _generate_correct_examples(self) -> str:
        """Generate constraint-aware CORRECT examples based on current min/max limits"""
        agent_config = getattr(self.config, self.role)
        role_term = "production" if self.role == 'manufacturer' else "orders"
        action = "producing" if self.role == 'manufacturer' else "ordering"

        if getattr(agent_config, 'no_decision_constraints', False):
            return (
                f'{{{{"decision": 5, "reason": "ST: net +3, current period OK. LT: coverage 0.8 — rebuilding pipeline."}}}}\n'
                f'{{{{"decision": 0, "reason": "ST: net +12, ample inventory. LT: coverage 1.5 — letting pipeline drain."}}}}\n'
                f'{{{{"decision": 8, "reason": "ST: net -2, immediate shortage. LT: coverage 0.6 — urgent replenishment needed."}}}}'
            )

        min_q = getattr(agent_config, 'min_order_quantity', None)
        max_q = getattr(agent_config, 'max_order_quantity', None)
        min_val = min_q if min_q is not None else 0
        max_val = max_q if max_q is not None else 20
        mid_val = (min_val + max_val) // 2
        low_val = max(min_val, min_val + (max_val - min_val) // 4)
        zero_allowed = (min_q is None or min_q == 0)

        examples = []
        if zero_allowed:
            examples.append(f'{{{{"decision": 0, "reason": "overstock — pausing {role_term} at minimum (0)"}}}}')
        else:
            examples.append(f'{{{{"decision": {min_val}, "reason": "overstock — {action} at minimum ({min_val})"}}}}')
        examples.append(f'{{{{"decision": {max_val}, "reason": "shortage — {action} at maximum ({max_val})"}}}}')
        examples.append(f'{{{{"decision": {mid_val}, "reason": "balanced — {action} near midpoint ({mid_val})"}}}}')
        if zero_allowed and low_val > 0:
            examples.append(f'{{{{"decision": {low_val}, "reason": "sufficient inventory — light {role_term} at low end ({low_val})"}}}}')
        elif not zero_allowed and low_val > min_val:
            examples.append(f'{{{{"decision": {low_val}, "reason": "sufficient inventory — light {role_term} near minimum ({low_val})"}}}}')

        return "\n".join(examples)

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
        hist_window = context.get('hist_window', 2 * lt)
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

        # Historical data (fetch early — needed for balance calculations)
        demand_history = context.get('demand_history', [])
        recent_orders = context.get('recent_orders', [])
        recent_inventory = context.get('recent_inventory', [])
        recent_costs = context.get('recent_costs', [])
        shipment_history = context.get('shipment_history', [])

        # Constraint range (extract early, used by both balance sections)
        agent_config = getattr(self.config, self.role)
        no_constraints = getattr(self.config, self.role).no_decision_constraints
        min_q = getattr(agent_config, 'min_order_quantity', None)
        max_q = getattr(agent_config, 'max_order_quantity', None)
        min_str = str(min_q) if min_q is not None else "0"
        max_str = str(max_q) if max_q is not None else "unlimited"

        # === Short-term Supply-Demand Balance (current period) ===
        net_inventory = inventory - backorders
        short_term_supply = net_inventory + next_shipment
        short_term_gap = short_term_supply - demand
        st_status = "ADEQUATE" if short_term_gap >= demand else ("TIGHT" if short_term_gap >= 0 else "SHORTAGE")
        prompt += f"""\n\n=== SHORT-TERM Balance (Current Period) ===
- Net inventory (inv - backorder): {net_inventory} units
- Next period arrival: {next_shipment} units
- Available supply (net inv + next arrival): {short_term_supply} units
- Current period demand: {demand} units
- Short-term gap: {short_term_gap:+d} units — Status: {st_status}"""
        if not no_constraints:
            prompt += f"""
- Your order constraint: [{min_str}, {max_str}]
- {'>> WARNING: Immediate shortage — backorder costs will be incurred! Consider ordering near MAX ({}) <<'.format(max_str) if short_term_gap < 0 else '>> Short-term position manageable — lean toward MIN ({}) if inventory is adequate <<'.format(min_str) if short_term_gap >= demand else 'Short-term position is tight — consider ordering above midpoint.'}"""
        else:
            prompt += f"""
- {'>> WARNING: Immediate shortage — backorder costs will be incurred! Order enough to cover the gap. <<' if short_term_gap < 0 else '>> Short-term position manageable — no urgency to order. <<' if short_term_gap >= demand else 'Short-term position is tight — consider ordering to rebuild.'}"""

        # === Long-term Supply-Demand Balance (over lead time horizon) ===
        long_term_supply = inventory - backorders + total_in_transit
        avg_demand = sum(demand_history) / len(demand_history) if demand_history else demand
        expected_demand_lt = avg_demand * lt if lt > 0 else avg_demand
        coverage_ratio = long_term_supply / expected_demand_lt if expected_demand_lt > 0 else float('inf')

        # Constraint-aware order guidance

        if no_constraints:
            if coverage_ratio > 1.5:
                lt_assessment = "OVERSURPLUS — reduce orders to prevent bullwhip amplification"
            elif coverage_ratio > 1.2:
                lt_assessment = "ADEQUATE — slight surplus, bias toward lower order quantities rather than maintaining current level"
            elif coverage_ratio >= 1.0:
                lt_assessment = "BALANCED — supply matches expected demand, maintain current ordering level"
            else:
                lt_assessment = "UNDERSUPPLY — increase orders to rebuild pipeline"
        else:
            if coverage_ratio > 1.5:
                lt_assessment = f"OVERSURPLUS — order at MINIMUM ({min_str}) to prevent bullwhip amplification"
            elif coverage_ratio > 1.2:
                lt_assessment = f"ADEQUATE — slight surplus, bias toward minimum ({min_str}) rather than maintaining current level"
            elif coverage_ratio >= 1.0:
                lt_assessment = f"BALANCED — supply matches expected demand, order near midpoint of [{min_str}, {max_str}]"
            else:
                lt_assessment = f"UNDERSUPPLY — order at MAXIMUM ({max_str}) to rebuild pipeline"

        prompt += f"""\n\n=== LONG-TERM Balance (Over {lt}-Period Lead Time Horizon) ===
- Net position (inventory - backorders): {inventory - backorders} units
- Total supply over lead time: net position + total {'WIP' if self.role == 'manufacturer' else 'in-transit'} ({total_in_transit}) = {long_term_supply} units
- Average demand per period: {avg_demand:.1f} units
- Expected demand over {lt} periods: {expected_demand_lt:.1f} units
- Coverage ratio: {coverage_ratio:.2f} — Assessment: {lt_assessment}"""
        if not no_constraints:
            prompt += f"""
- Your order constraint: [{min_str}, {max_str}]"""
        prompt += f"""
- Target coverage ratio: 1.1-1.3 (safety buffer for demand variability)"""
        if no_constraints:
            prompt += f"""
- {'>> ACTION: Order more aggressively to cover the gap <<' if coverage_ratio < 1.0 else '>> ACTION: Reduce orders to let pipeline drain <<' if coverage_ratio > 1.5 else 'Long-term balance is within acceptable range — maintain current trajectory.'}"""
        else:
            prompt += f"""
- {'>> ACTION: Order at MINIMUM ({}) <<'.format(min_str) if coverage_ratio > 1.5 else '>> ACTION: Order at MAXIMUM ({}) <<'.format(max_str) if coverage_ratio < 1.0 else 'Long-term balance is within acceptable range — choose the lowest cost option within your constraint range.'}"""

        # === Integrated Dual-Horizon Decision Guidance ===
        holding_cost_val = context.get('holding_cost', 0.5)
        backorder_cost_val = context.get('backorder_cost', 1.0)
        cost_ratio = backorder_cost_val / holding_cost_val if holding_cost_val > 0 else 2.0

        # Determine combined horizon status
        if short_term_gap < 0 and coverage_ratio < 1.0:
            horizon_scenario = "BOTH SHORTFALL — Both horizons indicate scarcity"
            if no_constraints:
                horizon_guidance = (
                    f">> URGENT: Short-term shortage + long-term undersupply. Must order significantly, "
                    f"covering both current-period demand and lead-time pipeline rebuild. "
                    f"This is the highest-priority scenario — do NOT under-order."
                )
            else:
                horizon_guidance = (
                    f">> URGENT: Short-term shortage + long-term undersupply. Must order MAX ({max_str}), "
                    f"covering both current-period demand and lead-time pipeline rebuild. "
                    f"This is the highest-priority scenario — do NOT under-order."
                )
        elif short_term_gap >= demand and coverage_ratio > 1.5:
            horizon_scenario = "BOTH SURPLUS — Both horizons indicate excess"
            if no_constraints:
                horizon_guidance = (
                    f">> RELAX: Short-term surplus + long-term oversupply. Order low, "
                    f"letting the pipeline drain naturally. Avoid adding any unnecessary inventory."
                )
            else:
                horizon_guidance = (
                    f">> RELAX: Short-term surplus + long-term oversupply. Order MIN ({min_str}), "
                    f"letting the pipeline drain naturally. Avoid adding any unnecessary inventory."
                )
        elif short_term_gap < 0 and coverage_ratio > 1.2:
            horizon_scenario = "CONFLICT: Short-term shortage vs long-term surplus"
            if no_constraints:
                horizon_guidance = (
                    f">> PRIORITIZE SHORT-TERM: Backorder cost = {cost_ratio:.1f}x holding cost, "
                    f"short-term takes priority. Order only enough to cover the gap (backorder + demand deficit), do NOT over-order. "
                    f"Target: choose a low quantity; clear backorders only — do NOT rebuild excess pipeline."
                )
            else:
                horizon_guidance = (
                    f">> PRIORITIZE SHORT-TERM: Backorder cost = {cost_ratio:.1f}x holding cost, "
                    f"short-term takes priority. Order only enough to cover the gap (backorder + demand deficit), do NOT order MAX. "
                    f"Target: choose in the lower half of [{min_str}, {max_str}]; clear backorders only — do NOT rebuild excess pipeline."
                )
        elif short_term_gap >= 0 and coverage_ratio < 1.0:
            horizon_scenario = "CONFLICT: Short-term surplus vs long-term shortage"
            if no_constraints:
                horizon_guidance = (
                    f">> REBUILD GRADUALLY: Short-term is adequate, but lead-time pipeline is too thin. "
                    f"Moderately increase orders to rebuild pipeline — do NOT panic-order, which would amplify the bullwhip effect. "
                    f"Target: choose a moderate quantity; gradually restore coverage to 1.1–1.3."
                )
            else:
                horizon_guidance = (
                    f">> REBUILD GRADUALLY: Short-term is adequate, but lead-time pipeline is too thin. "
                    f"Moderately increase orders to rebuild pipeline — do NOT panic-order MAX, which would amplify the bullwhip effect. "
                    f"Target: choose in the upper half of [{min_str}, {max_str}]; gradually restore coverage to 1.1–1.3."
                )
        else:
            horizon_scenario = "ALIGNED — Both horizons balanced"
            if no_constraints:
                horizon_guidance = (
                    f"Both horizons are within acceptable range. Maintain current trajectory; "
                    f"choose the lowest-cost option."
                )
            else:
                horizon_guidance = (
                    f"Both horizons are within acceptable range. Maintain current trajectory; "
                    f"choose the lowest-cost option within [{min_str}, {max_str}]."
                )

        prompt += f"""\n\n{'='*20}
=== INTEGRATED DECISION GUIDANCE ===
{'='*20}

DUAL-HORIZON ASSESSMENT: {horizon_scenario}

Short-Term: {st_status} (gap = {short_term_gap:+d} units)
Long-Term: Coverage = {coverage_ratio:.2f} (target 1.1–1.3)

DECISION PRIORITY: {horizon_guidance}

COST TRADEOFF: Backorder cost ({backorder_cost_val}/unit) = {cost_ratio:.1f}x holding cost ({holding_cost_val}/unit)
When the two horizons conflict, backorder cost carries higher weight, but do not sacrifice long-term stability — that creates bullwhip cycles.
"""
        if not no_constraints:
            prompt += f"""
ORDER CONSTRAINT: [{min_str}, {max_str}]
"""
        prompt += "\n"

        # Historical trend info

        if recent_orders:
            prompt += f"""\n\n=== Historical Trend Info ===
- Last {hist_window} periods order history: {recent_orders}
- Last {hist_window} periods inventory history: {recent_inventory}
- Last {hist_window} periods cost history: {[f'{c:.1f}' for c in recent_costs] if recent_costs else []}
- Last {hist_window} periods demand history: {demand_history}
- Last {hist_window} periods shipment history: {shipment_history}"""

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
                recent_retailer_demand = sum(retailer_demand_history[-hist_window:]) if len(retailer_demand_history) >= hist_window else total_retailer_demand
                prompt += f"- Retailer historical total demand: {total_retailer_demand} units ({len(retailer_demand_history)} periods total)\n"
                prompt += f"- Retailer average demand: {avg_retailer_demand:.1f} units/period\n"
                prompt += f"- Retailer last {hist_window} periods demand: {recent_retailer_demand} units\n"
            else:
                prompt += f"- Current period retailer demand: {current_retail_demand} units\n"
            prompt += f"- Current period retailer demand: {current_retail_demand} units\n"
            prompt += f"\nIMPORTANT: This is the market's real demand. Total orders across all supply chain participants should stay balanced with it!\n"


            # 2. My total order status analysis - clarify own ordering situation
            my_recent_orders = sum(context.get('recent_orders', [])[-hist_window:]) if context.get('recent_orders') else 0
            my_total_orders = sum(context.get('orders_history', [])) if context.get('orders_history') else 0
            my_current_order = context.get('current_order', 0)
            my_pending_orders = context.get('pending_orders', [])
            my_total_in_transit = context.get('total_in_transit', 0)

            prompt += f"\n\n[CORE] My Total Order Status Analysis:\n"
            prompt += f"- My current period order: {my_current_order} units\n"
            prompt += f"- My last {hist_window} periods total orders: {my_recent_orders} units\n"
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

        # Add order quantity limit requirements (skipped when no_constraints)
        if not no_constraints:
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

        # === Final dual-horizon mandate ===
        prompt += f"""\n\n{'='*20}
=== FINAL MANDATE: DUAL-HORIZON BALANCE ===
Before outputting your JSON decision, verify each item:
1. [ ] Short-Term Balance: Can you meet current-period demand without stockout? Is net position >= 0?
2. [ ] Long-Term Balance: Is coverage ratio in the 1.1–1.3 target range? Is the pipeline reasonable?
3. [ ] Horizon Reconciliation: If short-term and long-term conflict, did you apply the correct priority?
4. [ ] Cost Minimization: Does the decision minimize total cost (holding + backorder) across both horizons?
5. [ ] Bullwhip Check: Will my order amplify upstream demand variability? Am I overreacting?
"""

        return prompt

    def get_last_reasoning(self) -> str:
        """Get the reasoning process of the last decision"""
        return self.last_reasoning

    def get_decision_details(self) -> dict:
        """Get decision details including reasoning process and decision result"""
        return {
            'reasoning': self.last_reasoning,
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

            # Save complete response as raw response (for later analysis)
            self.last_raw_response = response

            # Attempt to extract JSON portion from response (handling cases with reasoning content)
            cleaned_response = self._extract_json_from_response(response)
            print(f"Extracted JSON portion: '{cleaned_response}'")

            # Prioritize JSON format parsing
            order_quantity, reason = self._parse_json_decision_with_reason(cleaned_response)

            # If JSON parsing succeeded, save the reason explanation
            if order_quantity is not None and reason is not None:
                self.last_decision_reason = reason
                self.last_decision_explanation = reason
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
        """Extract JSON portion from response"""
        import re

        # Method 1: Find any JSON object
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, response)
        if json_matches:
            for match in json_matches:
                if 'decision' in match or 'order_decision' in match or 'order_quantity' in match:
                    print(f"DEBUG: Found JSON object: {match}")
                    return match
            print(f"DEBUG: Using first JSON object: {json_matches[0]}")
            return json_matches[0]

        # Method 2: No JSON found, return raw response
        print(f"DEBUG: No JSON object found, returning raw response")
        return response

    def _extract_number_from_response(self, response: str) -> int:
        """Extract order quantity from LLM response using tiered fallback strategies"""
        import re

        print(f"DEBUG: [{self.role}] Starting number extraction...")

        # Method 1: Standalone numbers (surrounded by spaces/punctuation/line boundaries)
        standalone_numbers = re.findall(r'(?:^|\s|[^\d])(\d+)(?:\s|[^\d]|$)', response)
        print(f"Method 1: standalone numbers {standalone_numbers}")
        if standalone_numbers:
            valid_numbers = [int(n) for n in standalone_numbers if 0 <= int(n) <= 1000]
            if valid_numbers:
                preferred_numbers = [n for n in valid_numbers if 3 <= n <= 100]
                if preferred_numbers:
                    result = preferred_numbers[-1]
                    print(f"Method 1 success: preferred number {result}")
                    return result
                else:
                    result = valid_numbers[-1]
                    print(f"Method 1 success: standalone number {result}")
                    return result

        # Method 2: All numbers with smart filtering
        all_numbers = re.findall(r'\d+', response)
        print(f"Method 2: all numbers {all_numbers}")
        if all_numbers:
            int_numbers = [int(n) for n in all_numbers]
            filtered_numbers = []
            for num in int_numbers:
                if 0 <= num <= 1000:
                    if num not in [0, 1, 2] or len([x for x in int_numbers if x == num]) == 1:
                        filtered_numbers.append(num)

            if filtered_numbers:
                result = filtered_numbers[-1]
                print(f"Method 2 success: filtered number {result}")
                return result
            elif int_numbers:
                result = int_numbers[-1]
                print(f"Method 2 fallback: last number {result}")
                return min(max(result, 0), 1000)

        # Method 3: Keyword-based patterns
        keyword_patterns = [
            r'we need.*?(\d+)',
            r'i recommend.*?(\d+)',
            r'order.*?(\d+)',
            r'quantity.*?(\d+)',
            r'decision.*?(\d+)'
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, response.lower())
            if match:
                result = int(match.group(1))
                if 0 <= result <= 1000:
                    print(f"Method 3 success: keyword match {result}")
                    return result

        # No numbers found, return default
        print(f"WARNING: [{self.role}] Cannot extract any reasonable number from response, using default 0")
        print(f"WARNING: [{self.role}] Raw response: {response}")
        return 0

    def _apply_order_limits(self, order_quantity: int) -> int:
        """Apply order quantity limits"""
        agent_config = getattr(self.config, self.role)
        original_quantity = order_quantity
        no_constraints = getattr(agent_config, 'no_decision_constraints', False)

        # When no_decision_constraints is True, use wide safety bounds [0, 1000]
        # instead of the configured min/max (which may still be the defaults 3/8).
        if no_constraints:
            order_quantity = max(0, min(1000, order_quantity))
            if original_quantity != order_quantity:
                print(f"DEBUG: No-constraint role clamped to [0, 1000]: original={original_quantity}, adjusted={order_quantity}")
            return order_quantity

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

            # Create prompts
            system_prompt = self._create_system_prompt()

            # Append demand forecasting deduction fragment if enabled (add-on only)
            if getattr(self.config, 'enable_demand_forecasting', False):
                system_prompt += (
                    "\n\n"
                    "Enable Supply Chain Demand Forecasting Deduction Mode: "
                    "First, clearly identify your role in the supply chain. "
                    "Based on global behaviors and current state of all participants, "
                    "actively predict downstream and full supply chain short-term and mid-term demand fluctuation trends. "
                    "Must structurally record: demand forecast rationale, direction of change, affected nodes, and potential trajectories. "
                    "Maintain a dedicated standalone block for forecast records, "
                    "and iteratively revise predictions in each subsequent round. "
                    "This is an add-on output only — do NOT alter the original response structure."
                )

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

            # Save reasoning: prefer decision reason, fall back to raw response, then summary
            self.last_reasoning = (
                getattr(self, 'last_decision_reason', '') or
                self.last_raw_response or
                f"Decision: {final_order_quantity}"
            )

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
