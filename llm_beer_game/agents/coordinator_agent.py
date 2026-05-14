from typing import Dict, List, Any, Optional
import json
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass
from config.game_config import GameConfig
from agents.base_agent import BaseAgent

# Import analyzer
from analysis.decision_analyzer import DecisionAnalyzer

@dataclass
class GlobalSupplyChainState:
    """Global supply chain state"""
    current_round: int
    total_system_cost: float
    demand_trend: str  # "rising", "falling", "stable", "volatile"
    bullwhip_severity: str  # "severe", "moderate", "mild", "none"
    inventory_imbalance: Dict[str, str]  # Inventory status per role: "high", "normal", "low"
    coordination_priority: str  # "cost_control", "demand_response", "inventory_balance"

class CoordinatorAgent:
    """Supply chain coordinator agent

    Responsible for global supply chain state analysis, generating systematic decision guidance,
    and providing coordination suggestions to each role in mind map form.
    """

    def __init__(self,
                 config: GameConfig,
                 llm_client: Any,
                 agent_id: str = "coordinator"):
        """
        Initialize coordinator agent

        Args:
            config: Game configuration
            llm_client: LLM client
            agent_id: Agent ID
        """
        self.config = config
        self.llm_client = llm_client
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"Coordinator.{agent_id}")

        # Historical analysis data
        self.analysis_history: List[Dict[str, Any]] = []
        self.coordination_history: List[str] = []

        # Current global state
        self.current_global_state: Optional[GlobalSupplyChainState] = None

        # Last raw guidance data
        self._last_raw_guidance: Dict[str, Dict[str, Any]] = {}

        # Initialize decision analyzer
        if DecisionAnalyzer:
            self.decision_analyzer = DecisionAnalyzer()
        else:
            self.decision_analyzer = None

    def analyze_supply_chain(self,
                           round_data: Dict[str, Any],
                           agents_info: Dict[str, Dict[str, Any]],
                           previous_prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Analyze supply chain global state

        Args:
            round_data: Current round data
            agents_info: State information for each agent
            previous_prompts: Previous round prompts for each role

        Returns:
            Global analysis result
        """
        self.logger.info(f"Starting global supply chain analysis - Round {round_data.get('round', 0)}")

        # Collect global information
        global_context = self._collect_global_context(round_data, agents_info, previous_prompts)


        # Add decision quality analysis
        decision_analysis = self._analyze_decisions_quality(agents_info, round_data.get('round', 0))
        global_context['decision_analysis'] = decision_analysis

        # Generate analysis prompt
        analysis_prompt = self._create_analysis_prompt(global_context)

        # Call LLM for global analysis
        try:
            response = self.llm_client.generate(
                system_prompt="You are the global coordinator of the supply chain system.",
                user_prompt=analysis_prompt,
                temperature=0.1,
                max_tokens=800
            )

            # Parse analysis result
            analysis_result = self._parse_analysis_response(response)

            # Update global state
            self._update_global_state(analysis_result, round_data)

            # Record analysis history
            self.analysis_history.append({
                'round': round_data.get('round', 0),
                'analysis': analysis_result,
                'timestamp': round_data.get('timestamp')
            })

            self.logger.info(f"Global analysis complete - Coordination priority: {analysis_result.get('coordination_priority', 'N/A')}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Global analysis failed: {str(e)}")
            return self._fallback_analysis(global_context)

    def generate_coordination_guidance(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate coordination guidance for each role

        Args:
            analysis_result: Global analysis result

        Returns:
            Coordination guidance text for each role
        """
        guidance_prompt = self._create_guidance_prompt(analysis_result)

        try:
            response = self.llm_client.generate(
                system_prompt="You are a supply chain coordination expert.",
                user_prompt=guidance_prompt,
                temperature=0.1,
                max_tokens=1000
            )

            # Parse guidance for each role
            guidance = self._parse_guidance_response(response)

            # Record coordination history
            self.coordination_history.append(response)

            return guidance

        except Exception as e:
            self.logger.error(f"Coordination guidance generation failed: {str(e)}")
            return self._fallback_guidance(analysis_result)

    def generate_mindmap_markdown(self, analysis_result: Dict[str, Any]) -> str:
        """
        Generate mind map in Markdown format

        Args:
            analysis_result: Global analysis result

        Returns:
            Mind map Markdown text
        """
        # Per user request, no longer generate mind maps
        return ""

    def _collect_global_context(self,
                              round_data: Dict[str, Any],
                              agents_info: Dict[str, Dict[str, Any]],
                              previous_prompts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Collect global context information"""
        context = {
            'current_round': round_data.get('round', 0),
            'customer_demand': round_data.get('customer_demand', 0),
            'agents_state': {},
            'system_metrics': {},
            'previous_prompts': previous_prompts or {}
        }

        # Collect agent states
        total_inventory = 0
        total_cost = 0

        for role, agent_info in agents_info.items():
            state = agent_info.get('state', {})
            context['agents_state'][role] = {
                'inventory': state.get('inventory', 0),
                'backorder': state.get('backorder', 0),
                'total_cost': state.get('total_cost', 0),
                'recent_orders': state.get('orders_history', [])[-3:],
                'recent_demands': state.get('demand_history', [])[-3:]
            }
            total_inventory += state.get('inventory', 0)
            total_cost += state.get('total_cost', 0)

        # System-level metrics
        context['system_metrics'] = {
            'total_inventory': total_inventory,
            'total_cost': total_cost,
            'avg_inventory_per_agent': total_inventory / len(agents_info) if agents_info else 0
        }

        return context

    def _create_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create global analysis prompt"""
        # Check for custom analysis prompt
        if hasattr(self.config, 'coordinator_analysis_prompt') and self.config.coordinator_analysis_prompt:
            # Use custom prompt and fill in context
            prompt = self.config.coordinator_analysis_prompt

            # Build role state information
            agents_state_info = ""
            for role, state in context['agents_state'].items():
                role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler',
                            'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}.get(role, role)
                agents_state_info += f"""
{role_name}:
  - Inventory: {state['inventory']}, Backorder: {state['backorder']}
  - Total cost: {state['total_cost']:.2f}
  - Recent orders: {state['recent_orders']}
  - Recent demands: {state['recent_demands']}"""

            # Fill template variables
            prompt = prompt.replace('{current_round}', str(context['current_round']))
            prompt = prompt.replace('{customer_demand}', str(context['customer_demand']))
            prompt = prompt.replace('{system_metrics.total_inventory}', str(context['system_metrics']['total_inventory']))
            prompt = prompt.replace('{system_metrics.total_cost:.2f}', f"{context['system_metrics']['total_cost']:.2f}")
            prompt = prompt.replace('{agents_state_info}', agents_state_info.strip())

            return prompt

        # Use default prompt
        prompt = f"""You are coordinating the classic "Beer Game" supply chain scenario and must analyze and judge from a "bullwhip effect" perspective.

Key premise:
- The only controllable decision is the "ordering quantity" (ordering decision).
- It is strictly prohibited to propose strategies other than ordering (such as price adjustments, promotions, changing lead times/capacity/transportation, etc.).

Current Supply Chain State (Round {context['current_round']}):
Customer demand: {context['customer_demand']}
System total inventory: {context['system_metrics']['total_inventory']}
System total cost: {context['system_metrics']['total_cost']:.2f}

Detailed Role Statuses:
"""

        for role, state in context['agents_state'].items():
            role_name = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler',
                        'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}.get(role, role)
            prompt += f"""
{role_name}:
  - Inventory: {state['inventory']}, Backorder: {state['backorder']}
  - Total cost: {state['total_cost']:.2f}
  - Recent orders: {state['recent_orders']}
  - Recent demands: {state['recent_demands']}"""

        if context.get('previous_prompts'):
            prompt += "\n\nPrevious Round Decision Prompt Summary:\n"
            for role, prev_prompt in context['previous_prompts'].items():
                prompt += f"- {role}: {prev_prompt[:100]}...\n"

        # Add overreaction analysis information
        if context.get('overreaction_analysis'):
            overreaction = context['overreaction_analysis']
            prompt += f"\n\nOverreaction Risk Analysis:\n"
            prompt += f"System risk level: {overreaction.get('system_risk_level', 'unknown')}\n"

            if overreaction.get('bullwhip_contributors'):
                contributors = ', '.join(overreaction['bullwhip_contributors'])
                prompt += f"Bullwhip effect main contributors: {contributors}\n"

            for role, analysis in overreaction.get('agents_overreaction', {}).items():
                risk_level = analysis.get('risk_level', 'unknown')
                if risk_level in ['high', 'medium']:
                    prompt += f"- {role}: {risk_level} risk - {analysis.get('recommendation', '')}\n"

        # Add decision quality analysis information
        if context.get('decision_analysis'):
            decision_info = context['decision_analysis']
            prompt += "\n\nDecision Quality Analysis:\n"
            prompt += f"System coordination: {decision_info.get('system_coordination', 'unknown')}\n"

            agents_quality = decision_info.get('agents_quality', {})
            for agent, quality_info in agents_quality.items():
                quality_score = quality_info.get('quality_score', 0.5)
                issues = quality_info.get('issues', [])
                suggestions = quality_info.get('suggestions', [])

                prompt += f"- {agent}: Decision quality score {quality_score:.2f}\n"
                if issues:
                    prompt += f"  Issues: {', '.join(issues)}\n"
                if suggestions:
                    prompt += f"  Suggestions: {', '.join(suggestions)}\n"

            improvement_suggestions = decision_info.get('improvement_suggestions', [])
            if improvement_suggestions:
                prompt += f"System improvement suggestions: {', '.join(improvement_suggestions)}\n"

        prompt += """

Focus on the bullwhip effect as the core, paying special attention to order stability issues. Analyze the supply chain state from the following dimensions:

1. **Demand Trend Analysis**: Current customer demand change trend (rising/falling/stable/volatile)
2. **Bullwhip Effect Diagnosis**: Amplification of order fluctuations across the supply chain (severe/moderate/mild/none)
3. **Inventory Balance Assessment**: Whether inventory levels at each stage are reasonable (high/normal/low)
4. **Order Stability Check**: Whether each role's orders are continuous and stable, whether extreme decisions exist
5. **Bottleneck Identification**: The most critical problem areas in the supply chain

Focus on analysis, do not output JSON format diagnostic results. Your analysis will be used later to generate specific ordering guidance suggestions.
"""

        return prompt

    def _create_guidance_prompt(self, analysis_result: Dict[str, Any]) -> str:
        """Create coordination guidance prompt"""
        # Check for custom guidance prompt
        if hasattr(self.config, 'coordinator_guidance_prompt') and self.config.coordinator_guidance_prompt:
            # Use custom prompt and fill in analysis result
            prompt = self.config.coordinator_guidance_prompt

            # Fill template variables
            prompt = prompt.replace('{bullwhip_severity}', str(analysis_result.get('bullwhip_severity', 'N/A')))
            prompt = prompt.replace('{demand_trend}', str(analysis_result.get('demand_trend', 'N/A')))
            prompt = prompt.replace('{inventory_retailer}', str(analysis_result.get('inventory_imbalance', {}).get('retailer', 'unknown')))
            prompt = prompt.replace('{inventory_wholesaler}', str(analysis_result.get('inventory_imbalance', {}).get('wholesaler', 'unknown')))
            prompt = prompt.replace('{inventory_distributor}', str(analysis_result.get('inventory_imbalance', {}).get('distributor', 'unknown')))
            prompt = prompt.replace('{inventory_manufacturer}', str(analysis_result.get('inventory_imbalance', {}).get('manufacturer', 'unknown')))
            prompt = prompt.replace('{key_issues}', ', '.join(analysis_result.get('key_issues', [])))

            return prompt

        # Get inventory status and key issues
        inventory_status = analysis_result.get('inventory_imbalance', {})
        bullwhip_severity = analysis_result.get('bullwhip_severity', 'N/A')
        key_issues = analysis_result.get('key_issues', [])

        prompt = f"""You are the global coordinator of the Beer Game supply chain. Based on bullwhip effect analysis results, provide specific ordering guidance for supply chain roles that need intervention.

**Core Task**: Based on bullwhip effect diagnosis, provide specific order quantity recommendations for supply chain roles needing intervention.

**Analysis Conclusions**:
- Bullwhip effect severity: {bullwhip_severity}
- Demand trend: {analysis_result.get('demand_trend', 'N/A')}
- Inventory status analysis:
  * Retailer: {inventory_status.get('retailer', 'unknown')}
  * Wholesaler: {inventory_status.get('wholesaler', 'unknown')}
  * Distributor: {inventory_status.get('distributor', 'unknown')}
  * Manufacturer: {inventory_status.get('manufacturer', 'unknown')}
- Identified key issues: {', '.join(key_issues) if key_issues else 'None'}

**Guiding Principles**:
1. **Precise Intervention**: Only provide guidance for roles that truly need adjustment; roles in good condition do not need intervention
2. **Specific Quantity**: Directly provide clear order quantity recommendations, not vague adjustment directions
3. **Priority Ordering**: Handle the most severe problem areas first
4. **Cost Control**: Resolve issues while minimizing total costs
5. **System Optimization**: Consider overall supply chain coordination, avoid local optimization harming the whole

**Output Requirements**:
Provide JSON format specific guidance for each role needing intervention, including:
- intervention_required: true (must be true, indicating intervention needed)
- order_quantity: Directly provide specific order quantity (must be a positive integer)
- reason: Brief and clear reason in English (20 characters max)
- priority: high/medium/low (severity)

For roles not needing intervention, return:
- intervention_required: false

Return strict JSON format:
{{
  "retailer": {{
    "intervention_required": true,
    "order_quantity": 8,
    "reason": "Inventory low, restock",
    "priority": "high"
  }},
  "wholesaler": {{
    "intervention_required": false
  }},
  "distributor": {{
    "intervention_required": true,
    "order_quantity": 12,
    "reason": "Smooth order volatility",
    "priority": "medium"
  }},
  "manufacturer": {{
    "intervention_required": false
  }}
}}
"""

        return prompt

    def _create_mindmap_prompt(self, analysis_result: Dict[str, Any]) -> str:
        """Create mind map prompt"""
        # Check for custom mind map prompt
        if hasattr(self.config, 'coordinator_mindmap_prompt') and self.config.coordinator_mindmap_prompt:
            # Use custom prompt and fill in analysis result
            prompt = self.config.coordinator_mindmap_prompt

            # Get inventory status analysis
            inventory_status = analysis_result.get('inventory_imbalance', {})

            # Fill template variables
            prompt = prompt.replace('{demand_trend}', str(analysis_result.get('demand_trend', 'N/A')))
            prompt = prompt.replace('{bullwhip_severity}', str(analysis_result.get('bullwhip_severity', 'N/A')))
            prompt = prompt.replace('{inventory_status}', str(inventory_status))
            prompt = prompt.replace('{key_issues}', ', '.join(analysis_result.get('key_issues', [])))
            prompt = prompt.replace('{strategic_recommendations}', ', '.join(analysis_result.get('strategic_recommendations', [])))
            prompt = prompt.replace('{inventory_retailer}', str(inventory_status.get('retailer', 'unknown')))
            prompt = prompt.replace('{inventory_wholesaler}', str(inventory_status.get('wholesaler', 'unknown')))
            prompt = prompt.replace('{inventory_distributor}', str(inventory_status.get('distributor', 'unknown')))
            prompt = prompt.replace('{inventory_manufacturer}', str(inventory_status.get('manufacturer', 'unknown')))

            return prompt

        # Get inventory status analysis
        inventory_status = analysis_result.get('inventory_imbalance', {})

        prompt = f"""Based on Beer Game simulation results, generate a bullwhip effect analysis mind map.

**Current Simulation Data**:
- Demand trend: {analysis_result.get('demand_trend', 'N/A')}
- Bullwhip effect severity: {analysis_result.get('bullwhip_severity', 'N/A')}
- Role inventory status: {inventory_status}
- Key issues: {analysis_result.get('key_issues', [])}
- Strategic recommendations: {analysis_result.get('strategic_recommendations', [])}

Generate Markdown format bullwhip effect analysis mind map, **focusing on the following structure**:

# Beer Game Bullwhip Effect Diagnosis Report

## Bullwhip Effect Overall Assessment
### Severity Determination
- Based on current data: {analysis_result.get('bullwhip_severity', 'N/A')}
- Fluctuation amplification analysis
- Cost impact assessment

### Supply Chain Stability
- Demand transmission distortion
- Inventory fluctuation range
- Order coefficient of variation

## Key Problem Localization Analysis
### Retailer Issues
- Inventory status: {inventory_status.get('retailer', 'unknown')}
- Order decision deviation
- Demand forecast accuracy

### Wholesaler Issues
- Inventory status: {inventory_status.get('wholesaler', 'unknown')}
- Information transmission delay
- Order amplification effect

### Distributor Issues
- Inventory status: {inventory_status.get('distributor', 'unknown')}
- Intermediate coordination
- Inventory buffer strategy

### Manufacturer Issues
- Inventory status: {inventory_status.get('manufacturer', 'unknown')}
- Production plan fluctuation
- Capacity utilization efficiency

### Order Stability Analysis
- Order continuity assessment for each role
- Extreme order decision identification
- Long-term no-order or excessive order issues
- Impact of order fluctuation on bullwhip effect

## Targeted Solutions
### Retailer Optimization Strategy
- **Specific Prompt Guidance**: Adjust orders based on actual demand, avoid panic ordering
- **Order Stability Requirements**: Maintain order continuity, avoid long periods without ordering or sudden large orders
- Demand forecast improvement methods
- Inventory management optimization

### Wholesaler Optimization Strategy
- **Specific Prompt Guidance**: Smooth order transmission, reduce fluctuation amplification
- **Order Stability Requirements**: Keep orders relatively stable, avoid extreme order decisions
- Information sharing mechanism
- Order batch optimization

### Distributor Optimization Strategy
- **Specific Prompt Guidance**: Maintain stable supply rhythm, avoid overreaction
- **Order Stability Requirements**: Ensure order continuity, avoid stockouts or excessive hoarding
- Inventory buffer management
- Coordination mechanism establishment

### Manufacturer Optimization Strategy
- **Specific Prompt Guidance**: Focus on long-term demand trends, establish stable production plans
- **Order Stability Requirements**: Maintain order and production continuity, avoid production stoppages or overproduction
- Capacity planning optimization
- Long-term trend analysis

### Order Stability Improvement Measures
- **Continuity Principle**: Ensure each role's orders have continuity and predictability
- **Extreme Avoidance**: Prevent extreme behavior of long periods without orders or sudden large orders
- **Gradual Adjustment**: Adopt small, gradual order quantity adjustment strategy
- **Coordinated Consistency**: Keep order decisions coordinated across roles, avoid local extremes
- Supply chain visualization

## Systemic Improvement Measures
### Information Transparency
### Coordination Mechanism Optimization
### Incentive Mechanism Design
### Risk Sharing Strategy

**Requirements**:
1. Must analyze based on actual simulation data
2. Clearly identify which of the four enterprises has the most severe issues
3. Provide specific prompt improvement suggestions for each role
4. Solutions must be actionable
"""

        return prompt

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse analysis response"""
        json_str = None
        try:
            # Attempt JSON parsing
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]

                # Clean JSON string
                json_str = json_str.strip()
                # Remove possible code block markers
                json_str = json_str.replace('```json', '').replace('```', '')
                # Remove excess whitespace and newlines
                json_str = ' '.join(json_str.split())

                # Attempt multiple fix strategies
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # First attempt: basic fix
                            fixed_json = self._fix_common_json_errors(json_str)
                        elif attempt == 1:
                            # Second attempt: more aggressive fix
                            fixed_json = self._aggressive_json_fix(json_str)
                        else:
                            # Third attempt: simplified JSON structure
                            fixed_json = self._simplify_json_structure(json_str)

                        self.logger.debug(f"JSON fix attempt {attempt+1}: {fixed_json[:200]}...")
                        parsed_data = json.loads(fixed_json)

                        # Validate required fields and provide defaults
                        required_fields = {
                            'demand_trend': 'stable',
                            'bullwhip_severity': 'mild',
                            'inventory_imbalance': {
                                'retailer': 'normal',
                                'wholesaler': 'normal',
                                'distributor': 'normal',
                                'manufacturer': 'normal'
                            },
                            'coordination_priority': 'inventory_balance',
                            'key_issues': ['More data analysis needed'],
                            'strategic_recommendations': ['Continue monitoring supply chain state']
                        }

                        # Fill missing fields
                        for field, default_value in required_fields.items():
                            if field not in parsed_data:
                                parsed_data[field] = default_value

                        self.logger.info(f"JSON parsing succeeded (attempt {attempt+1})")
                        return parsed_data

                    except json.JSONDecodeError as je:
                        self.logger.debug(f"JSON parsing attempt {attempt+1} failed: {je}")
                        if attempt == 2:  # Last attempt failed
                            self.logger.warning(f"All JSON parsing attempts failed, last error: {je}")
                            break
                        continue

        except Exception as e:
            self.logger.warning(f"Failed to parse analysis response: {e}")
            self.logger.debug(f"Raw response: {response[:500]}...")
            self.logger.debug(f"Processed JSON string: {json_str[:200] if json_str else 'N/A'}...")

        # Fallback parsing
        self.logger.info("Using default analysis result")
        return {
            'demand_trend': 'stable',
            'bullwhip_severity': 'mild',
            'inventory_imbalance': {
                'retailer': 'normal',
                'wholesaler': 'normal',
                'distributor': 'normal',
                'manufacturer': 'normal'
            },
            'coordination_priority': 'inventory_balance',
            'key_issues': ['JSON parsing failed, need to check LLM response format'],
            'strategic_recommendations': ['Continue monitoring supply chain state, optimize LLM prompts']
        }

    def _parse_guidance_response(self, response: str) -> Dict[str, str]:
        """Parse guidance response (compatible with both strings and structured objects)"""
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]

                # Log raw JSON string for debugging
                self.logger.debug(f"Raw JSON string: {json_str[:200]}...")
                self.logger.debug(f"JSON string length: {len(json_str)}")

                # Basic cleanup and fault tolerance
                json_str = json_str.strip().replace('```json', '').replace('```', '')
                json_str = ' '.join(json_str.split())

                self.logger.debug(f"Cleaned JSON string: {json_str[:200]}...")

                # Attempt multiple fix strategies
                parsed = {}  # Initialize parsed variable
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # First attempt: basic fix
                            self.logger.debug(f"Attempting basic JSON fix...")
                            fixed_json = self._fix_common_json_errors(json_str)
                        elif attempt == 1:
                            # Second attempt: more aggressive fix
                            self.logger.debug(f"Attempting aggressive JSON fix...")
                            fixed_json = self._aggressive_json_fix(json_str)
                        else:
                            # Third attempt: simplified JSON structure
                            self.logger.debug(f"Attempting simplified JSON structure...")
                            fixed_json = self._simplify_json_structure(json_str)

                        self.logger.debug(f"Fixed JSON (attempt {attempt+1}): {fixed_json[:200]}...")
                        parsed = json.loads(fixed_json)
                        self.logger.info(f"JSON parsing succeeded (attempt {attempt+1})")
                        break
                    except json.JSONDecodeError as je:
                        self.logger.debug(f"JSON parsing attempt {attempt+1} failed: {je}")
                        error_pos = getattr(je, 'pos', 19)
                        # Initialize fixed_json variable
                        fixed_json = json_str
                        if hasattr(je, 'pos') and je.pos < len(fixed_json):
                            error_context = fixed_json[max(0, je.pos-20):je.pos+20]
                            self.logger.debug(f"Error context: ...{error_context}...")
                        if attempt == 2:
                            # All attempts failed, return default parse result
                            self.logger.warning(f"All JSON parsing attempts failed, using default results")
                            parsed = {
                                'retailer': {'intervention_required': False},
                                'wholesaler': {'intervention_required': False},
                                'distributor': {'intervention_required': False},
                                'manufacturer': {'intervention_required': False}
                            }
                            break
                        continue

                roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
                out: Dict[str, str] = {}
                raw_guidance: Dict[str, Dict[str, Any]] = {}  # Store raw guidance data
                # Ensure parsed is defined
                if not parsed:
                    parsed = {}
                for r in roles:
                    if r in parsed:
                        val = parsed[r]
                        if isinstance(val, dict):
                            # Store raw guidance data
                            raw_guidance[r] = val
                            # Check if intervention needed
                            if val.get('intervention_required', False):
                                out[r] = self._format_guidance_obj(val)
                            else:
                                # No intervention needed, return empty string
                                out[r] = ""
                        elif isinstance(val, str):
                            # If it's a non-empty string, use it
                            guidance_str = val.strip()
                            if guidance_str:
                                out[r] = guidance_str
                            else:
                                out[r] = ""
                        else:
                            out[r] = ""
                    else:
                        # Role not in parsed results, provide no guidance
                        out[r] = ""

                # Store raw guidance data in instance variable for external access
                self._last_raw_guidance = raw_guidance
                return out

        except Exception as e:
            self.logger.warning(f"Failed to parse guidance response: {e}")
            self.logger.debug(f"Full response content: {response}")

        # Fallback guidance - return empty strings when no intervention needed
        return {
            'retailer': '',
            'wholesaler': '',
            'distributor': '',
            'manufacturer': ''
        }

    def _update_global_state(self, analysis_result: Dict[str, Any], round_data: Dict[str, Any]):
        """Update global state"""
        self.current_global_state = GlobalSupplyChainState(
            current_round=round_data.get('round', 0),
            total_system_cost=round_data.get('total_cost', 0),
            demand_trend=analysis_result.get('demand_trend', 'stable'),
            bullwhip_severity=analysis_result.get('bullwhip_severity', 'mild'),
            inventory_imbalance=analysis_result.get('inventory_imbalance', {}),
            coordination_priority=analysis_result.get('coordination_priority', 'inventory_balance')
        )

    def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis"""
        return {
            'demand_trend': 'stable',
            'bullwhip_severity': 'mild',
            'inventory_imbalance': {
                'retailer': 'normal',
                'wholesaler': 'normal',
                'distributor': 'normal',
                'manufacturer': 'normal'
            },
            'coordination_priority': 'inventory_balance',
            'key_issues': ['System operating normally'],
            'strategic_recommendations': ['Continue current strategy']
        }

    def _fallback_guidance(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Fallback guidance"""
        # When no intervention needed, provide no guidance
        return {
            'retailer': '',
            'wholesaler': '',
            'distributor': '',
            'manufacturer': ''
        }

    def _fix_common_json_errors(self, json_str: str) -> str:
        """Fix common JSON format errors"""
        import re

        # Record original string for debugging
        original_str = json_str

        try:
            # Basic cleanup
            json_str = json_str.strip()

            # If string is empty or too short, return default JSON directly
            if len(json_str) < 2:
                # Check if it's a guidance response
                if 'intervention_required' in original_str:
                    return json.dumps({
                        'retailer': {'intervention_required': False},
                        'wholesaler': {'intervention_required': False},
                        'distributor': {'intervention_required': False},
                        'manufacturer': {'intervention_required': False}
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        'demand_trend': 'stable',
                        'bullwhip_severity': 'mild',
                        'coordination_priority': 'inventory_balance'
                    }, ensure_ascii=False)

            # Remove code block markers
            json_str = json_str.replace('```json', '').replace('```', '')

            # Find first { and last }, keep only the JSON portion in between
            start_brace = json_str.find('{')
            end_brace = json_str.rfind('}')

            if start_brace == -1 or end_brace == -1 or start_brace >= end_brace:
                # No valid JSON structure found, return default JSON
                self.logger.debug("No valid JSON structure found, using default structure")
                # Check if it's a guidance response
                if 'intervention_required' in original_str:
                    return json.dumps({
                        'retailer': {'intervention_required': False},
                        'wholesaler': {'intervention_required': False},
                        'distributor': {'intervention_required': False},
                        'manufacturer': {'intervention_required': False}
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        'demand_trend': 'stable',
                        'bullwhip_severity': 'mild',
                        'coordination_priority': 'inventory_balance'
                    }, ensure_ascii=False)

            # Extract JSON portion
            json_str = json_str[start_brace:end_brace + 1]

            # Preprocessing: fix common issues
            json_str = json_str.replace("'", '"')  # Convert single quotes to double quotes
            json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace

            # Fix key quoting issues (safer regex)
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

            # Fix value quoting issues - only string values
            def fix_value_quotes(match):
                prefix = match.group(1)  # :
                value = match.group(2).strip()  # value
                suffix = match.group(3)  # ,}]

                # If value already has quotes or is a number/boolean, don't modify
                if (value.startswith('"') and value.endswith('"')) or \
                   value in ['true', 'false', 'null'] or \
                   value.isdigit() or \
                   (value.startswith('-') and value[1:].isdigit()) or \
                   value.startswith('{') or value.startswith('['):
                    return f'{prefix}{value}{suffix}'

                # Otherwise add quotes
                return f'{prefix}"{value}"{suffix}'

            json_str = re.sub(r'(:\s*)([^,}\]]+?)([,}\]])', fix_value_quotes, json_str)

            # Fix trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            # Fix missing comma issues
            # 1. Fix missing comma after string values
            json_str = re.sub(r'(")\s+(")' , r'\1,\2', json_str)

            # 2. Fix missing comma after object endings
            json_str = re.sub(r'(})\s+(")', r'\1,\2', json_str)

            # 3. Fix missing comma after array endings
            json_str = re.sub(r'(\])\s+(")', r'\1,\2', json_str)

            # 4. Fix missing comma after numeric values
            json_str = re.sub(r'(\d)\s+(")', r'\1,\2', json_str)

            # 5. Fix missing comma after boolean values
            json_str = re.sub(r'(true|false|null)\s+(")', r'\1,\2', json_str)

            # 6. Fix comma issues in nested structures
            json_str = re.sub(r'(["\d}\]])\s+(["\w])', r'\1,\2', json_str)

            # 7. Fix consecutive comma issues
            json_str = re.sub(r',+', ',', json_str)

            # 8. Fix trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            # 9. Fix double quote repetition
            json_str = re.sub(r'""([^"]*?)""', r'"\1"', json_str)

            # Final validation
            try:
                parsed = json.loads(json_str)
                self.logger.debug(f"JSON fix succeeded: {json_str[:100]}...")
                return json_str
            except json.JSONDecodeError as e:
                error_pos = getattr(e, 'pos', 19)
                error_context = json_str[max(0, error_pos-10):error_pos+10] if error_pos < len(json_str) else "N/A"
                self.logger.warning(f"JSON still invalid after fix: {e}, position: {error_pos}")
                self.logger.debug(f"Error context: ...{error_context}...")

                # If still failing, try more aggressive fix
                return self._create_fallback_json(original_str)

        except Exception as e:
            self.logger.error(f"Error during JSON fix process: {e}")
            self.logger.debug(f"Raw JSON: {original_str[:200]}...")
            return self._create_fallback_json(original_str)

    def _create_fallback_json(self, original_str: str) -> str:
        """Create fallback JSON structure"""
        try:
            # Attempt to extract some key information from the original text
            fallback_data = {
                'demand_trend': 'stable',
                'bullwhip_severity': 'mild',
                'coordination_priority': 'inventory_balance',
                'key_issues': ['JSON parsing failed'],
                'strategic_recommendations': ['Use default strategy']
            }

            # Simple keyword extraction
            if 'rising' in original_str.lower() or 'increasing' in original_str.lower() or '上升' in original_str:
                fallback_data['demand_trend'] = 'rising'
            elif 'falling' in original_str.lower() or 'decreasing' in original_str.lower() or '下降' in original_str:
                fallback_data['demand_trend'] = 'falling'

            if 'severe' in original_str.lower() or '严重' in original_str:
                fallback_data['bullwhip_severity'] = 'severe'

            return json.dumps(fallback_data, ensure_ascii=False)
        except:
            # Last backup solution
            return '{"demand_trend": "stable", "bullwhip_severity": "mild", "coordination_priority": "inventory_balance"}'

    def _aggressive_json_fix(self, json_str: str) -> str:
        """More aggressive JSON fix strategy"""
        import re

        try:
            # Remove all newlines and excess spaces
            json_str = re.sub(r'\s+', ' ', json_str.strip())

            # Remove possible non-JSON content
            json_str = re.sub(r'^[^{]*', '', json_str)  # Remove non-JSON content at start
            json_str = re.sub(r'[^}]*$', '', json_str)  # Remove non-JSON content at end

            # Ensure basic JSON structure
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str = json_str + '}'

            # Fix common format issues
            json_str = json_str.replace('""', '"')  # Fix double quote repetition
            json_str = json_str.replace(',,', ',')   # Fix comma repetition
            json_str = json_str.replace(': ,', ': "",')
            json_str = json_str.replace(':,', ': "",')

            # Fix key quoting issues
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

            # Fix value quoting issues - more conservative approach
            json_str = re.sub(r'(:\s*)([^"\[\{\d\-][^,}\]]*?)([,}\]])',
                            lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}'
                            if m.group(2).strip() and m.group(2).strip() not in ['true', 'false', 'null']
                            else f'{m.group(1)}{m.group(2).strip() or ""}{m.group(3)}',
                            json_str)

            # Fix trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

            # Fix missing commas - improved version, better handling of complex cases
            # Add comma after value when missing
            json_str = re.sub(r'(["\d}\]])\s+(["\w])', r'\1,\2', json_str)

            # Handle missing comma after objects or arrays
            json_str = re.sub(r'(})\s*(["\w])', r'\1,\2', json_str)
            json_str = re.sub(r'(\])\s*(["\w])', r'\1,\2', json_str)

            # Fix consecutive comma issues
            json_str = re.sub(r',+', ',', json_str)

            # Final cleanup
            json_str = json_str.strip()

            # Validate fix result
            try:
                json.loads(json_str)
                self.logger.debug(f"Aggressive JSON fix succeeded: {json_str[:100]}...")
                return json_str
            except json.JSONDecodeError as e:
                self.logger.debug(f"Aggressive JSON fix still failed: {e}")
                return self._create_fallback_json(json_str)

        except Exception as e:
            self.logger.debug(f"Aggressive JSON fix process exception: {e}")
            return self._create_fallback_json(json_str)

        return json_str

    def _simplify_json_structure(self, json_str: str) -> str:
        import re

        # Check if it's a guidance response JSON structure
        if 'intervention_required' in json_str:
            # Create a minimal valid guidance JSON structure
            simplified: Dict[str, Any] = {
                'retailer': {'intervention_required': False},
                'wholesaler': {'intervention_required': False},
                'distributor': {'intervention_required': False},
                'manufacturer': {'intervention_required': False}
            }

            try:
                # Attempt to extract useful information from original string
                # Extract roles needing intervention
                roles_patterns = [
                    r'"(retailer|wholesaler|distributor|manufacturer)"\s*:\s*\{[^}]*"intervention_required"\s*:\s*true',
                ]
                for pattern in roles_patterns:
                    matches = re.findall(pattern, json_str, re.IGNORECASE)
                    for role in matches:
                        # Attempt to extract order quantity
                        order_pattern = rf'"{role}"\s*:\s*{{[^}}]*"order_quantity"\s*:\s*(\d+)'
                        order_match = re.search(order_pattern, json_str, re.IGNORECASE)
                        order_quantity = int(order_match.group(1)) if order_match else 8

                        # Attempt to extract reason
                        reason_pattern = rf'"{role}"\s*:\s*{{[^}}]*"reason"\s*:\s*"([^"]*)"'
                        reason_match = re.search(reason_pattern, json_str, re.IGNORECASE)
                        reason = reason_match.group(1) if reason_match else "System recommended adjustment"

                        # Attempt to extract priority
                        priority_pattern = rf'"{role}"\s*:\s*{{[^}}]*"priority"\s*:\s*"([^"]*)"'
                        priority_match = re.search(priority_pattern, json_str, re.IGNORECASE)
                        priority = priority_match.group(1) if priority_match else "medium"

                        simplified[role] = {
                            'intervention_required': True,
                            'order_quantity': order_quantity,
                            'reason': reason,
                            'priority': priority
                        }

            except Exception as e:
                self.logger.debug(f"Failed to extract information when simplifying guidance JSON structure: {str(e)}")

            return json.dumps(simplified, ensure_ascii=False)
        else:
            # Create a minimal valid analysis JSON structure
            simplified_analysis: Dict[str, Any] = {
                'demand_trend': 'stable',
                'bullwhip_severity': 'mild',
                'inventory_imbalance': {
                    'retailer': 'normal',
                    'wholesaler': 'normal',
                    'distributor': 'normal',
                    'manufacturer': 'normal'
                },
                'coordination_priority': 'inventory_balance',
                'key_issues': ['JSON format error, using simplified structure'],
                'strategic_recommendations': ['Continue monitoring supply chain state']
            }

            try:
                # Attempt to extract useful information from original string
                # Extract demand trend
                trend_patterns = [
                    r'demand_trend["\s]*:["\s]*([^,}"]+)',
                    r'需求趋势["\s]*:["\s]*([^,}"]+)',
                    r'趋势["\s]*:["\s]*([^,}"]+)'
                ]
                for pattern in trend_patterns:
                    match = re.search(pattern, json_str, re.IGNORECASE)
                    if match:
                        trend = match.group(1).strip().strip('"').strip("'")
                        if trend in ['rising', 'falling', 'stable', 'volatile', '上升', '下降', '稳定', '波动']:
                            trend_map = {'上升': 'rising', '下降': 'falling', '稳定': 'stable', '波动': 'volatile'}
                            simplified_analysis['demand_trend'] = trend_map.get(trend, trend)
                        break

                # Extract bullwhip effect severity
                bullwhip_patterns = [
                    r'bullwhip_severity["\s]*:["\s]*([^,}"]+)',
                    r'牛鞭效应["\s]*:["\s]*([^,}"]+)'
                ]
                for pattern in bullwhip_patterns:
                    match = re.search(pattern, json_str, re.IGNORECASE)
                    if match:
                        severity = match.group(1).strip().strip('"').strip("'")
                        if severity in ['severe', 'moderate', 'mild', 'none', '严重', '中等', '轻微', '无']:
                            severity_map = {'严重': 'severe', '中等': 'moderate', '轻微': 'mild', '无': 'none'}
                            simplified_analysis['bullwhip_severity'] = severity_map.get(severity, severity)
                        break

                # Extract coordination priority
                priority_patterns = [
                    r'coordination_priority["\s]*:["\s]*([^,}"]+)',
                    r'协调优先级["\s]*:["\s]*([^,}"]+)'
                ]
                for pattern in priority_patterns:
                    match = re.search(pattern, json_str, re.IGNORECASE)
                    if match:
                        priority = match.group(1).strip().strip('"').strip("'")
                        if priority in ['cost_control', 'demand_response', 'inventory_balance', '成本控制', '需求响应', '库存平衡']:
                            priority_map = {'成本控制': 'cost_control', '需求响应': 'demand_response', '库存平衡': 'inventory_balance'}
                            simplified_analysis['coordination_priority'] = priority_map.get(priority, priority)
                        break

            except Exception as e:
                self.logger.debug(f"Failed to extract information when simplifying JSON structure: {str(e)}")

            return json.dumps(simplified_analysis, ensure_ascii=False)

    def _fallback_mindmap(self, analysis_result: Dict[str, Any]) -> str:

        return """# Supply Chain Coordination Analysis
## Current Status
- System operating normally
- All roles coordinating well

## Suggestions
- Continue current strategy
- Maintain system stability
"""

    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get coordination summary"""
        if self.current_global_state:
            return {
                'total_analyses': len(self.analysis_history),
                'current_state': self.current_global_state.__dict__,
                'recent_priorities': [analysis.get('analysis', {}).get('coordination_priority')
                                   for analysis in self.analysis_history[-5:]]
            }
        else:
            return {
                'total_analyses': len(self.analysis_history),
                'current_state': None,
                'recent_priorities': [analysis.get('analysis', {}).get('coordination_priority')
                                   for analysis in self.analysis_history[-5:]]
            }
        """Get coordination summary"""
        return {
            'total_analyses': len(self.analysis_history),
            'current_state': self.current_global_state.__dict__ if self.current_global_state else None,
            'recent_priorities': [analysis.get('analysis', {}).get('coordination_priority')
                               for analysis in self.analysis_history[-5:]]
        }

    def _format_guidance_obj(self, obj: Dict[str, Any]) -> str:
        """Convert structured guidance object to concise, executable one-line English instruction"""
        # Check if intervention needed
        if not obj.get('intervention_required', False):
            return ""  # No intervention needed, return empty string

        # Directly return specific order quantity and reason
        order_quantity = obj.get('order_quantity')
        reason = obj.get('reason', '')

        if isinstance(order_quantity, (int, float)) and reason:
            return f"Please order {int(order_quantity)} units, {reason}"
        elif isinstance(order_quantity, (int, float)):
            return f"Please order {int(order_quantity)} units"
        elif reason:
            return reason
        else:
            return "Maintain stable ordering, avoid overreaction"



    def _analyze_decisions_quality(self, agents_info: Dict[str, Dict[str, Any]], round_num: int) -> Dict[str, Any]:
        """Analyze decision quality for each agent"""
        try:
            decision_analysis = {
                'agents_quality': {},
                'system_coordination': 'good',
                'improvement_suggestions': []
            }

            poor_decision_count = 0

            for role, agent_info in agents_info.items():
                agent_state = agent_info.get('state', {})

                # Get historical data
                orders_history = agent_state.get('orders_history', [])
                demand_history = agent_state.get('demand_history', [])
                inventory = agent_state.get('inventory', 0)
                backorder = agent_state.get('backorder', 0)

                if len(orders_history) >= 2:
                    # Use decision analyzer (if available)
                    if self.decision_analyzer:
                        decision_data = {
                            'orders': orders_history[-5:],
                            'demands': demand_history[-5:],
                            'inventory': inventory,
                            'backorder': backorder
                        }
                        quality_metrics = self.decision_analyzer.analyze_decision_quality(
                            agent_name=role,
                            decision_data=decision_data
                        )
                    else:
                        # Simple decision quality assessment
                        quality_metrics = self._simple_decision_quality_assessment(
                            role, orders_history, demand_history, inventory, backorder
                        )

                    decision_analysis['agents_quality'][role] = quality_metrics

                    if quality_metrics.get('quality_score', 0.5) < 0.3:
                        poor_decision_count += 1
                else:
                    decision_analysis['agents_quality'][role] = {
                        'quality_score': 0.5,
                        'issues': ['Insufficient data'],
                        'suggestions': ['Need more historical data for analysis']
                    }

            # Assess system coordination
            if poor_decision_count >= 2:
                decision_analysis['system_coordination'] = 'poor'
                decision_analysis['improvement_suggestions'].append('Multiple agents have poor decision quality, systemic improvement needed')
            elif poor_decision_count == 1:
                decision_analysis['system_coordination'] = 'moderate'
                decision_analysis['improvement_suggestions'].append('Individual agents need to optimize decision strategies')

            return decision_analysis

        except Exception as e:
            self.logger.error(f"Decision quality analysis failed: {e}")
            return {
                'agents_quality': {},
                'system_coordination': 'unknown',
                'improvement_suggestions': ['Analysis failed, recommend maintaining current strategy']
            }

    def _simple_decision_quality_assessment(self, role: str, orders: List[int], demands: List[int],
                                          inventory: int, backorder: int) -> Dict[str, Any]:
        """Simple decision quality assessment"""
        issues = []
        suggestions = []
        quality_score = 0.7  # Default medium quality

        if len(orders) >= 2:
            # Check order volatility
            order_variance = sum((orders[i] - orders[i-1])**2 for i in range(1, len(orders))) / (len(orders) - 1)
            if order_variance > 100:  # High volatility
                issues.append('Excessive order volatility')
                suggestions.append('Reduce large order quantity fluctuations')
                quality_score -= 0.2

        # Check inventory status
        if inventory > 20:
            issues.append('Inventory too high')
            suggestions.append('Appropriately reduce order quantity')
            quality_score -= 0.1
        elif backorder > 10:
            issues.append('Severe backorders')
            suggestions.append('Increase order quantity to meet demand')
            quality_score -= 0.2

        # Check demand responsiveness
        if len(demands) >= 2 and len(orders) >= 2:
            demand_trend = demands[-1] - demands[-2] if len(demands) >= 2 else 0
            order_trend = orders[-1] - orders[-2] if len(orders) >= 2 else 0

            if demand_trend > 0 and order_trend <= 0:
                issues.append('Demand rising but orders not increasing')
                suggestions.append('Adjust ordering strategy based on demand trend')
                quality_score -= 0.1

        return {
            'quality_score': max(0.0, min(1.0, quality_score)),
            'issues': issues,
            'suggestions': suggestions
        }
