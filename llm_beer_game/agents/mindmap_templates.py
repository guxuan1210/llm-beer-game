"""Mind map decision templates

Provides structured mind map templates for MindmapDecisionAgent.
"""

from typing import Dict, Any, List

class MindmapTemplate:
    """Mind map template base class"""

    @staticmethod
    def get_decision_mindmap_template() -> str:
        """Get decision mind map base template"""
        return """
# Supply Chain Decision Mind Map

## 1. Current State Analysis
### Individual State
- Inventory level: {current_inventory}
- Backorder status: {backorder}
- Incoming shipment: {incoming_shipment}
- Current demand: {current_demand}

### Historical Trends
- Demand history: {demand_history}
- Order history: {order_history}
- Inventory changes: {inventory_trend}

## 2. Shared Information Integration
### Upstream/Downstream Status
{shared_info_section}

### System-Level Metrics
- Total inventory level: {total_inventory}
- Supply chain stability: {supply_chain_stability}
- Bullwhip effect level: {bullwhip_effect}

## 3. Risk Assessment
### Stockout Risk
- Current risk level: {stockout_risk}
- Influencing factors: {stockout_factors}

### Overstock Risk
- Current risk level: {overstock_risk}
- Cost impact: {holding_cost_impact}

## 4. Decision Option Analysis
### Option A: Conservative Order ({conservative_order})
- Advantages: {conservative_pros}
- Disadvantages: {conservative_cons}
- Risks: {conservative_risks}

### Option B: Moderate Order ({moderate_order})
- Advantages: {moderate_pros}
- Disadvantages: {moderate_cons}
- Risks: {moderate_risks}

### Option C: Aggressive Order ({aggressive_order})
- Advantages: {aggressive_pros}
- Disadvantages: {aggressive_cons}
- Risks: {aggressive_risks}

## 5. Coordination Suggestions Integration
{coordinator_guidance_section}

## 6. Final Decision Reasoning
### Key Considerations
1. {key_factor_1}
2. {key_factor_2}
3. {key_factor_3}

### Decision Logic
{decision_logic}

### Recommended Order Quantity
**{recommended_order}** (Based on above analysis)

### Decision Confidence
{confidence_level} - {confidence_reason}
"""

    @staticmethod
    def get_shared_info_template() -> str:
        """Get shared information template"""
        return """
### Retailer Status
- Inventory: {retailer_inventory}
- Demand: {retailer_demand}
- Order: {retailer_order}

### Wholesaler Status
- Inventory: {wholesaler_inventory}
- Demand: {wholesaler_demand}
- Order: {wholesaler_order}

### Distributor Status
- Inventory: {distributor_inventory}
- Demand: {distributor_demand}
- Order: {distributor_order}

### Manufacturer Status
- Inventory: {manufacturer_inventory}
- Production: {manufacturer_production}
- Order: {manufacturer_order}
"""

    @staticmethod
    def get_coordinator_guidance_template() -> str:
        """Get coordinator guidance template"""
        return """
### Global Analysis
{global_analysis}

### Systemic Recommendations
{systemic_recommendations}

### Coordination Focus
{coordination_focus}

### Risk Alerts
{risk_alerts}
"""

    @staticmethod
    def get_risk_assessment_template() -> str:
        """Get risk assessment template"""
        return """
### Short-term Risk (1-3 rounds)
- Stockout probability: {short_term_stockout_prob}
- Overstock probability: {short_term_overstock_prob}
- Cost impact: {short_term_cost_impact}

### Medium-term Risk (4-8 rounds)
- Demand fluctuation impact: {medium_term_demand_impact}
- Supply chain stability: {medium_term_stability}
- Adjustment flexibility: {medium_term_flexibility}

### Long-term Risk (9+ rounds)
- Systemic risk: {long_term_systemic_risk}
- Competitive impact: {long_term_competitive_impact}
- Strategic adjustment needs: {long_term_strategic_needs}
"""

class RoleSpecificTemplates:
    """Mind map template definitions for generating decision mind maps"""

    @staticmethod
    def get_retailer_template() -> str:
        """Retailer-specific template"""
        return """
# Retailer Decision Mind Map

## Customer Demand Analysis
### Current Demand Pattern
- Actual customer demand: {customer_demand}
- Demand trend: {demand_trend}
- Seasonal factors: {seasonal_factors}

### Customer Satisfaction Considerations
- Stockout impact: {stockout_customer_impact}
- Service level target: {service_level_target}
- Competitive pressure: {competitive_pressure}

## Upstream Supply Analysis
### Wholesaler Supply Capacity
- Supply stability: {wholesaler_supply_stability}
- Delivery reliability: {delivery_reliability}
- Supply risk: {supply_risk}

{base_template}
"""

    @staticmethod
    def get_wholesaler_template() -> str:
        """Wholesaler-specific template"""
        return """
# Wholesaler Decision Mind Map

## Retailer Demand Analysis
### Downstream Demand Aggregation
- Retailer order pattern: {retailer_order_pattern}
- Demand amplification effect: {demand_amplification}
- Order volatility: {order_volatility}

### Multi-Customer Balance
- Customer priority: {customer_priority}
- Allocation strategy: {allocation_strategy}
- Relationship management: {relationship_management}

## Upstream Coordination
### Distributor Coordination
- Supply coordination: {distributor_coordination}
- Information sharing effectiveness: {info_sharing_effectiveness}
- Joint forecasting: {joint_forecasting}

{base_template}
"""

    @staticmethod
    def get_distributor_template() -> str:
        """Distributor-specific template"""
        return """
# Distributor Decision Mind Map

## Midstream Coordination Analysis
### Wholesaler Demand Aggregation
- Wholesaler order pattern: {wholesaler_order_pattern}
- Regional demand differences: {regional_demand_differences}
- Demand forecast accuracy: {forecast_accuracy}

### Manufacturer Coordination
- Production plan coordination: {production_planning_coordination}
- Capacity utilization optimization: {capacity_utilization}
- Supply chain synchronization: {supply_chain_synchronization}

## Inventory Optimization
### Buffer Inventory Strategy
- Safety stock level: {safety_stock_level}
- Buffer design: {buffer_design}
- Risk diversification: {risk_diversification}

{base_template}
"""

    @staticmethod
    def get_manufacturer_template() -> str:
        """Manufacturer-specific template"""
        return """
# Manufacturer Decision Mind Map

## Production Plan Analysis
### Capacity Planning
- Current capacity utilization: {current_capacity_utilization}
- Production efficiency: {production_efficiency}
- Capacity bottlenecks: {capacity_bottlenecks}

### Production Cost Optimization
- Economies of scale effect: {economies_of_scale}
- Production cost structure: {production_cost_structure}
- Cost control opportunities: {cost_control_opportunities}

## Supply Chain Source Responsibility
### Demand Signal Processing
- Real demand identification: {real_demand_identification}
- Bullwhip mitigation: {bullwhip_mitigation}
- Demand smoothing strategy: {demand_smoothing}

### System Stability
- Supply chain stability maintenance: {supply_chain_stability_maintenance}
- Volatility control: {volatility_control}
- Long-term sustainability: {long_term_sustainability}

{base_template}
"""

class MindmapFormatter:
    """Mind map formatting tool"""

    @staticmethod
    def format_template(template: str, context: Dict[str, Any]) -> str:
        """Format mind map template

        Args:
            template: Template string
            context: Context data

        Returns:
            Formatted mind map
        """
        try:
            return template.format(**context)
        except KeyError as e:
            # If certain keys are missing, replace with default values
            safe_context = {}
            for key, value in context.items():
                safe_context[key] = str(value) if value is not None else "unknown"

            # Add default values
            defaults = {
                'current_inventory': 'unknown',
                'backorder': '0',
                'incoming_shipment': 'unknown',
                'current_demand': 'unknown',
                'shared_info_section': 'No shared information',
                'coordinator_guidance_section': 'No coordination guidance',
                'confidence_level': 'Medium',
                'confidence_reason': 'Based on currently available information'
            }

            for key, default_value in defaults.items():
                if key not in safe_context:
                    safe_context[key] = default_value

            try:
                return template.format(**safe_context)
            except KeyError:
                # If still problematic, return simplified version
                return f"# Decision Mind Map\n\nCurrent status: {safe_context.get('current_inventory', 'unknown')}\nDecision recommendation: Make decision based on available information"

    @staticmethod
    def create_decision_options(current_inventory: int, demand_forecast: float, role: str) -> Dict[str, Any]:
        """Create decision options

        Args:
            current_inventory: Current inventory
            demand_forecast: Demand forecast
            role: Role type

        Returns:
            Decision options dictionary
        """
        base_order = max(0, int(demand_forecast - current_inventory))

        conservative_order = max(0, base_order - 2)
        moderate_order = base_order
        aggressive_order = base_order + 3

        return {
            'conservative_order': conservative_order,
            'moderate_order': moderate_order,
            'aggressive_order': aggressive_order,
            'conservative_pros': 'Reduce inventory cost, lower overstock risk',
            'conservative_cons': 'May face stockout risk',
            'conservative_risks': 'Customer satisfaction decline',
            'moderate_pros': 'Balance inventory cost and service level',
            'moderate_cons': 'May not be flexible enough for changes',
            'moderate_risks': 'Medium risk level',
            'aggressive_pros': 'Ensure sufficient supply, improve service level',
            'aggressive_cons': 'Increase inventory holding costs',
            'aggressive_risks': 'Overstock risk'
        }
