import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from collections import deque

class OverreactionSeverity(Enum):
    """Overreaction severity"""
    NONE = "No Overreaction"
    MILD = "Mild Overreaction"
    MODERATE = "Moderate Overreaction"
    SEVERE = "Severe Overreaction"
    EXTREME = "Extreme Overreaction"

@dataclass
class OverreactionMetrics:
    """Overreaction metrics"""
    # Basic metrics
    amplification_ratio: float  # Amplification factor
    volatility_ratio: float     # Volatility ratio
    response_delay: int         # Response delay
    overcorrection_factor: float # Overcorrection factor

    # Advanced metrics
    bullwhip_contribution: float # Contribution to bullwhip effect
    demand_distortion: float     # Demand signal distortion
    stability_index: float       # Stability index

    # Classification results
    severity: OverreactionSeverity
    confidence: float           # Judgment confidence

    # Detailed analysis
    trigger_patterns: List[str] # Trigger patterns
    impact_assessment: Dict[str, float] # Impact assessment

class OverreactionAnalyzer:
    """Overreaction analyzer

    Used for detecting and quantifying overreaction behavior of supply chain
    participants, providing detailed analysis metrics and improvement suggestions.
    """

    def __init__(self, window_size: int = 5, sensitivity: float = 0.1):
        """
        Initialize overreaction analyzer

        Args:
            window_size: Analysis window size
            sensitivity: Sensitivity threshold
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.logger = logging.getLogger("OverreactionAnalyzer")

        # Historical data cache
        self.demand_history: Dict[str, deque] = {}
        self.order_history: Dict[str, deque] = {}
        self.inventory_history: Dict[str, deque] = {}

        # Analysis result cache
        self.analysis_cache: Dict[str, List[OverreactionMetrics]] = {}

        # Threshold configuration
        self.thresholds = {
            'amplification_mild': 1.2,
            'amplification_moderate': 1.5,
            'amplification_severe': 2.0,
            'amplification_extreme': 3.0,
            'volatility_mild': 1.3,
            'volatility_moderate': 1.8,
            'volatility_severe': 2.5,
            'volatility_extreme': 4.0,
            'stability_good': 0.8,
            'stability_fair': 0.6,
            'stability_poor': 0.4
        }

    def update_data(self, round_data: Dict[str, Any]):
        """Update analysis data

        Args:
            round_data: Current round data
        """
        try:
            # Extract role data
            agents_data = round_data.get('agents', {})
            customer_demand = round_data.get('customer_demand', 0)

            # Update demand history (customer demand faced by retailer)
            if 'retailer' not in self.demand_history:
                self.demand_history['retailer'] = deque(maxlen=self.window_size * 2)
            self.demand_history['retailer'].append(customer_demand)

            # Update order and inventory history for each role
            roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']

            for i, role in enumerate(roles):
                if role in agents_data:
                    agent_data = agents_data[role]

                    # Initialize history records
                    if role not in self.order_history:
                        self.order_history[role] = deque(maxlen=self.window_size * 2)
                        self.inventory_history[role] = deque(maxlen=self.window_size * 2)
                        self.analysis_cache[role] = []

                    # Update order history
                    order_placed = agent_data.get('order_placed', 0)
                    self.order_history[role].append(order_placed)

                    # Update inventory history
                    inventory = agent_data.get('end_state', {}).get('inventory', 0)
                    self.inventory_history[role].append(inventory)

                    # Update upstream demand history (except retailer)
                    if i > 0:  # Not retailer
                        upstream_role = roles[i-1]
                        if upstream_role not in self.demand_history:
                            self.demand_history[upstream_role] = deque(maxlen=self.window_size * 2)

                        # Upstream order is the demand faced by current role
                        if len(self.order_history[upstream_role]) > 0:
                            upstream_order = list(self.order_history[upstream_role])[-1]
                            self.demand_history[upstream_role].append(upstream_order)

        except Exception as e:
            self.logger.error(f"Error updating data: {e}")

    def analyze_agent_overreaction(self, role: str) -> Optional[OverreactionMetrics]:
        """Analyze single agent overreaction

        Args:
            role: Agent role

        Returns:
            Overreaction metrics, returns None if insufficient data
        """
        try:
            # Check data sufficiency
            if (role not in self.order_history or
                role not in self.demand_history or
                len(self.order_history[role]) < self.window_size or
                len(self.demand_history[role]) < self.window_size):
                return None

            # Get historical data
            orders = list(self.order_history[role])
            demands = list(self.demand_history[role])

            # Ensure consistent data length
            min_len = min(len(orders), len(demands))
            orders = orders[-min_len:]
            demands = demands[-min_len:]

            if min_len < self.window_size:
                return None

            # Calculate basic metrics
            amplification_ratio = self._calculate_amplification_ratio(orders, demands)
            volatility_ratio = self._calculate_volatility_ratio(orders, demands)
            response_delay = self._calculate_response_delay(orders, demands)
            overcorrection_factor = self._calculate_overcorrection_factor(orders, demands)

            # Calculate advanced metrics
            bullwhip_contribution = self._calculate_bullwhip_contribution(orders, demands)
            demand_distortion = self._calculate_demand_distortion(orders, demands)
            stability_index = self._calculate_stability_index(orders)

            # Determine severity
            severity, confidence = self._determine_severity(
                amplification_ratio, volatility_ratio, stability_index
            )

            # Identify trigger patterns
            trigger_patterns = self._identify_trigger_patterns(orders, demands)

            # Assess impact
            impact_assessment = self._assess_impact(
                role, amplification_ratio, volatility_ratio, bullwhip_contribution
            )

            # Create metrics object
            metrics = OverreactionMetrics(
                amplification_ratio=amplification_ratio,
                volatility_ratio=volatility_ratio,
                response_delay=response_delay,
                overcorrection_factor=overcorrection_factor,
                bullwhip_contribution=bullwhip_contribution,
                demand_distortion=demand_distortion,
                stability_index=stability_index,
                severity=severity,
                confidence=confidence,
                trigger_patterns=trigger_patterns,
                impact_assessment=impact_assessment
            )

            # Cache results
            self.analysis_cache[role].append(metrics)
            if len(self.analysis_cache[role]) > 10:  # Keep the last 10 analyses
                self.analysis_cache[role].pop(0)

            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing {role} overreaction: {e}")
            return None

    def analyze_supply_chain_overreaction(self) -> Dict[str, OverreactionMetrics]:
        """Analyze overreaction across the entire supply chain

        Returns:
            Overreaction metrics for each role
        """
        results = {}
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']

        for role in roles:
            metrics = self.analyze_agent_overreaction(role)
            if metrics:
                results[role] = metrics

        return results

    def _calculate_amplification_ratio(self, orders: List[float], demands: List[float]) -> float:
        """Calculate amplification ratio"""
        try:
            if len(orders) < 2 or len(demands) < 2:
                return 1.0

            # Calculate the magnitude of changes in orders and demands
            order_changes = [abs(orders[i] - orders[i-1]) for i in range(1, len(orders))]
            demand_changes = [abs(demands[i] - demands[i-1]) for i in range(1, len(demands))]

            avg_order_change = np.mean(order_changes) if order_changes else 0
            avg_demand_change = np.mean(demand_changes) if demand_changes else 0

            if avg_demand_change == 0:
                return 1.0 if avg_order_change == 0 else float('inf')

            return avg_order_change / avg_demand_change

        except Exception:
            return 1.0

    def _calculate_volatility_ratio(self, orders: List[float], demands: List[float]) -> float:
        """Calculate volatility ratio"""
        try:
            if len(orders) < 2 or len(demands) < 2:
                return 1.0

            order_volatility = np.std(orders) if len(orders) > 1 else 0
            demand_volatility = np.std(demands) if len(demands) > 1 else 0

            if demand_volatility == 0:
                return 1.0 if order_volatility == 0 else float('inf')

            return order_volatility / demand_volatility

        except Exception:
            return 1.0

    def _calculate_response_delay(self, orders: List[float], demands: List[float]) -> int:
        """Calculate response delay"""
        try:
            if len(orders) < 3 or len(demands) < 3:
                return 0

            # Find demand change points
            demand_changes = []
            for i in range(1, len(demands)):
                if abs(demands[i] - demands[i-1]) > self.sensitivity * np.mean(demands):
                    demand_changes.append(i)

            if not demand_changes:
                return 0

            # Find corresponding order response
            delays = []
            for change_point in demand_changes:
                for delay in range(1, min(4, len(orders) - change_point)):
                    response_point = change_point + delay
                    if response_point < len(orders):
                        if abs(orders[response_point] - orders[change_point]) > self.sensitivity * np.mean(orders):
                            delays.append(delay)
                            break

            return int(np.mean(delays)) if delays else 0

        except Exception:
            return 0

    def _calculate_overcorrection_factor(self, orders: List[float], demands: List[float]) -> float:
        """Calculate overcorrection factor"""
        try:
            if len(orders) < 3 or len(demands) < 3:
                return 1.0

            # Find demand changes and corresponding order responses
            overcorrections = []

            for i in range(1, min(len(orders), len(demands)) - 1):
                demand_change = demands[i] - demands[i-1]
                order_change = orders[i] - orders[i-1]

                if abs(demand_change) > self.sensitivity * np.mean(demands):
                    if demand_change != 0:
                        correction_ratio = order_change / demand_change
                        if abs(correction_ratio) > 1:
                            overcorrections.append(abs(correction_ratio))

            return np.mean(overcorrections) if overcorrections else 1.0

        except Exception:
            return 1.0

    def _calculate_bullwhip_contribution(self, orders: List[float], demands: List[float]) -> float:
        """Calculate contribution to bullwhip effect"""
        try:
            if len(orders) < 2 or len(demands) < 2:
                return 0.0

            # Calculate coefficient of variation for orders and demands
            order_cv = np.std(orders) / np.mean(orders) if np.mean(orders) > 0 else 0
            demand_cv = np.std(demands) / np.mean(demands) if np.mean(demands) > 0 else 0

            # Bullwhip effect contribution = order CV - demand CV
            contribution = max(0, order_cv - demand_cv)

            return contribution

        except Exception:
            return 0.0

    def _calculate_demand_distortion(self, orders: List[float], demands: List[float]) -> float:
        """Calculate demand signal distortion"""
        try:
            if len(orders) < 2 or len(demands) < 2:
                return 0.0

            # Calculate correlation between orders and demands
            if len(orders) == len(demands) and len(orders) > 1:
                correlation, _ = stats.pearsonr(orders, demands)
                # Distortion = 1 - correlation (lower correlation means higher distortion)
                distortion = max(0, 1 - abs(correlation))
                return distortion

            return 0.0

        except Exception:
            return 0.0

    def _calculate_stability_index(self, orders: List[float]) -> float:
        """Calculate stability index"""
        try:
            if len(orders) < 2:
                return 1.0

            # Calculate coefficient of variation of orders
            mean_order = np.mean(orders)
            if mean_order == 0:
                return 1.0

            cv = np.std(orders) / mean_order

            # Stability index = 1 / (1 + CV)
            stability = 1 / (1 + cv)

            return stability

        except Exception:
            return 1.0

    def _determine_severity(self, amplification_ratio: float,
                          volatility_ratio: float,
                          stability_index: float) -> Tuple[OverreactionSeverity, float]:
        """Determine overreaction severity"""
        try:
            # Calculate composite score
            amp_score = 0
            vol_score = 0
            stab_score = 0

            # Amplification ratio score
            if amplification_ratio >= self.thresholds['amplification_extreme']:
                amp_score = 4
            elif amplification_ratio >= self.thresholds['amplification_severe']:
                amp_score = 3
            elif amplification_ratio >= self.thresholds['amplification_moderate']:
                amp_score = 2
            elif amplification_ratio >= self.thresholds['amplification_mild']:
                amp_score = 1

            # Volatility score
            if volatility_ratio >= self.thresholds['volatility_extreme']:
                vol_score = 4
            elif volatility_ratio >= self.thresholds['volatility_severe']:
                vol_score = 3
            elif volatility_ratio >= self.thresholds['volatility_moderate']:
                vol_score = 2
            elif volatility_ratio >= self.thresholds['volatility_mild']:
                vol_score = 1

            # Stability score (reverse)
            if stability_index <= self.thresholds['stability_poor']:
                stab_score = 4
            elif stability_index <= self.thresholds['stability_fair']:
                stab_score = 3
            elif stability_index <= self.thresholds['stability_good']:
                stab_score = 2
            else:
                stab_score = 1

            # Composite score
            total_score = (amp_score + vol_score + stab_score) / 3

            # Determine severity
            if total_score >= 3.5:
                severity = OverreactionSeverity.EXTREME
                confidence = 0.9
            elif total_score >= 2.5:
                severity = OverreactionSeverity.SEVERE
                confidence = 0.8
            elif total_score >= 1.5:
                severity = OverreactionSeverity.MODERATE
                confidence = 0.7
            elif total_score >= 0.5:
                severity = OverreactionSeverity.MILD
                confidence = 0.6
            else:
                severity = OverreactionSeverity.NONE
                confidence = 0.5

            return severity, confidence

        except Exception:
            return OverreactionSeverity.NONE, 0.5

    def _identify_trigger_patterns(self, orders: List[float], demands: List[float]) -> List[str]:
        """Identify trigger patterns"""
        patterns = []

        try:
            if len(orders) < 3 or len(demands) < 3:
                return patterns

            # Detect sudden increase pattern
            for i in range(1, len(orders)):
                if orders[i] > orders[i-1] * 1.5:
                    patterns.append("Sudden large order increase")
                    break

            # Detect panic buying pattern
            consecutive_increases = 0
            for i in range(1, len(orders)):
                if orders[i] > orders[i-1]:
                    consecutive_increases += 1
                else:
                    consecutive_increases = 0

                if consecutive_increases >= 3:
                    patterns.append("Consecutive order increases (panic buying)")
                    break

            # Detect excessive inventory accumulation
            if len(self.inventory_history) > 0:
                role_inventories = list(self.inventory_history.values())
                if role_inventories:
                    recent_inventory = list(role_inventories[0])[-3:] if len(role_inventories[0]) >= 3 else []
                    if len(recent_inventory) >= 2 and all(recent_inventory[i] > recent_inventory[i-1] * 1.2 for i in range(1, len(recent_inventory))):
                        patterns.append("Excessive inventory accumulation")

            # Detect demand signal misinterpretation
            if len(orders) == len(demands):
                misaligned_responses = 0
                for i in range(1, len(orders)):
                    demand_trend = demands[i] - demands[i-1]
                    order_trend = orders[i] - orders[i-1]

                    # Demand decreasing but orders increasing, or demand increasing but orders over-increasing
                    if (demand_trend < 0 and order_trend > 0) or (demand_trend > 0 and order_trend > demand_trend * 2):
                        misaligned_responses += 1

                if misaligned_responses >= 2:
                    patterns.append("Demand signal misinterpretation")

            return patterns

        except Exception:
            return patterns

    def _assess_impact(self, role: str, amplification_ratio: float,
                      volatility_ratio: float, bullwhip_contribution: float) -> Dict[str, float]:
        """Assess impact"""
        impact = {
            "Cost Impact": 0.0,
            "Service Level Impact": 0.0,
            "Supply Chain Stability Impact": 0.0,
            "Downstream Propagation Impact": 0.0
        }

        try:
            # Cost impact (based on amplification ratio and volatility)
            cost_impact = min(1.0, (amplification_ratio - 1) * 0.3 + (volatility_ratio - 1) * 0.2)
            impact["Cost Impact"] = max(0.0, cost_impact)

            # Service level impact (overreaction may cause stockouts or excess inventory)
            service_impact = min(1.0, abs(amplification_ratio - 1) * 0.25)
            impact["Service Level Impact"] = service_impact

            # Supply chain stability impact
            stability_impact = min(1.0, bullwhip_contribution * 2)
            impact["Supply Chain Stability Impact"] = stability_impact

            # Downstream propagation impact (based on role position)
            role_weights = {
                'retailer': 0.8,    # Retailer impact relatively small
                'wholesaler': 1.0,  # Wholesaler impact moderate
                'distributor': 1.2, # Distributor impact relatively large
                'manufacturer': 1.5 # Manufacturer impact largest
            }

            downstream_impact = min(1.0, amplification_ratio * role_weights.get(role, 1.0) * 0.2)
            impact["Downstream Propagation Impact"] = downstream_impact

            return impact

        except Exception:
            return impact

    def get_improvement_suggestions(self, role: str, metrics: OverreactionMetrics) -> List[str]:
        """Get improvement suggestions

        Args:
            role: Agent role
            metrics: Overreaction metrics

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        try:
            # General suggestions based on severity
            if metrics.severity == OverreactionSeverity.EXTREME:
                suggestions.append("🚨 Extreme overreaction: Adjust decision strategy immediately, avoid extreme order decisions")
                suggestions.append("📊 Recommend implementing strict order review mechanism")
            elif metrics.severity == OverreactionSeverity.SEVERE:
                suggestions.append("⚠️ Severe overreaction: Need significant improvement in decision logic")
                suggestions.append("📈 Recommend improving demand forecast accuracy")
            elif metrics.severity == OverreactionSeverity.MODERATE:
                suggestions.append("📋 Moderate overreaction: Optimize order decision strategy")
            elif metrics.severity == OverreactionSeverity.MILD:
                suggestions.append("💡 Mild overreaction: Fine-tune decision parameters")

            # Suggestions based on specific metrics
            if metrics.amplification_ratio > 2.0:
                suggestions.append(f"📉 Order amplification ratio too high ({metrics.amplification_ratio:.2f}): Reduce sharp changes in order quantities")

            if metrics.volatility_ratio > 2.0:
                suggestions.append(f"📊 Order volatility too high ({metrics.volatility_ratio:.2f}): Adopt smoother order strategy")

            if metrics.stability_index < 0.5:
                suggestions.append(f"⚖️ Poor order stability ({metrics.stability_index:.2f}): Improve order decision consistency")

            if metrics.bullwhip_contribution > 0.3:
                suggestions.append(f"🌊 High bullwhip effect contribution ({metrics.bullwhip_contribution:.2f}): Reduce demand signal amplification propagation")

            # Role-specific suggestions
            role_specific_suggestions = {
                'retailer': [
                    "🛒 Focus on real customer demand, avoid panic buying",
                    "📱 Build customer demand forecasting model",
                    "🎯 Optimize safety stock level settings"
                ],
                'wholesaler': [
                    "🔄 Smooth upstream and downstream order transmission",
                    "📊 Aggregate demand information from multiple retailers",
                    "⚖️ Avoid amplifying downstream fluctuations to upstream"
                ],
                'distributor': [
                    "🎯 Identify real market demand signals",
                    "📈 Establish long-term demand trend analysis",
                    "🤝 Strengthen coordination and communication with manufacturers"
                ],
                'manufacturer': [
                    "🏭 Develop stable production plans",
                    "📊 Make decisions based on long-term trends rather than short-term fluctuations",
                    "⚙️ Optimize capacity utilization and inventory management"
                ]
            }

            if role in role_specific_suggestions:
                suggestions.extend(role_specific_suggestions[role])

            # Suggestions based on trigger patterns
            for pattern in metrics.trigger_patterns:
                if "Sudden large order increase" in pattern:
                    suggestions.append("🚫 Avoid sudden large order increases, adopt gradual adjustments")
                elif "panic buying" in pattern:
                    suggestions.append("😌 Stay calm, make decisions based on data rather than emotion")
                elif "Excessive inventory accumulation" in pattern:
                    suggestions.append("📦 Optimize inventory management, avoid excessive accumulation")
                elif "Demand signal misinterpretation" in pattern:
                    suggestions.append("🔍 Improve demand signal interpretation capability, avoid misjudgment")

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return ["📋 Recommend optimizing order decision strategy"]

    def generate_analysis_report(self, supply_chain_metrics: Dict[str, OverreactionMetrics]) -> str:
        """Generate analysis report

        Args:
            supply_chain_metrics: Overreaction metrics for each supply chain role

        Returns:
            Analysis report text
        """
        try:
            report = ["# Supply Chain Overreaction Analysis Report\n"]

            # Overview
            report.append("## 📊 Overview")

            total_agents = len(supply_chain_metrics)
            severe_count = sum(1 for m in supply_chain_metrics.values()
                             if m.severity in [OverreactionSeverity.SEVERE, OverreactionSeverity.EXTREME])

            report.append(f"- Number of agents analyzed: {total_agents}")
            report.append(f"- Agents with severe overreaction: {severe_count}")
            report.append(f"- Overall risk level: {'High' if severe_count > total_agents/2 else 'Medium' if severe_count > 0 else 'Low'}\n")

            # Detailed analysis by role
            report.append("## 🔍 Detailed Analysis by Role")

            for role, metrics in supply_chain_metrics.items():
                role_names = {
                    'retailer': 'Retailer',
                    'wholesaler': 'Wholesaler',
                    'distributor': 'Distributor',
                    'manufacturer': 'Manufacturer'
                }

                role_name = role_names.get(role, role)
                report.append(f"\n### {role_name}")
                report.append(f"- **Severity**: {metrics.severity.value} (Confidence: {metrics.confidence:.2f})")
                report.append(f"- **Amplification Ratio**: {metrics.amplification_ratio:.2f}")
                report.append(f"- **Volatility Ratio**: {metrics.volatility_ratio:.2f}")
                report.append(f"- **Stability Index**: {metrics.stability_index:.2f}")
                report.append(f"- **Bullwhip Effect Contribution**: {metrics.bullwhip_contribution:.2f}")

                if metrics.trigger_patterns:
                    report.append(f"- **Trigger Patterns**: {', '.join(metrics.trigger_patterns)}")

                # Impact assessment
                report.append("- **Impact Assessment**:")
                for impact_type, impact_value in metrics.impact_assessment.items():
                    report.append(f"  - {impact_type}: {impact_value:.2f}")

            # Improvement suggestions
            report.append("\n## 💡 Improvement Suggestions")

            for role, metrics in supply_chain_metrics.items():
                role_names = {
                    'retailer': 'Retailer',
                    'wholesaler': 'Wholesaler',
                    'distributor': 'Distributor',
                    'manufacturer': 'Manufacturer'
                }

                role_name = role_names.get(role, role)
                suggestions = self.get_improvement_suggestions(role, metrics)

                if suggestions:
                    report.append(f"\n### {role_name} Improvement Suggestions")
                    for suggestion in suggestions:
                        report.append(f"- {suggestion}")

            return "\n".join(report)

        except Exception as e:
            self.logger.error(f"Error generating analysis report: {e}")
            return "Analysis report generation failed"

    def reset(self):
        """Reset analyzer state"""
        self.demand_history.clear()
        self.order_history.clear()
        self.inventory_history.clear()
        self.analysis_cache.clear()
        self.logger.info("Overreaction analyzer has been reset")
