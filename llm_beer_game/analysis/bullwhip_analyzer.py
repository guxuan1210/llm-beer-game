from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Deferred import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llm_beer_game.simulation.game_engine import SimulationResult

@dataclass
class BullwhipMetrics:
    """Bullwhip effect metrics"""
    coefficient_of_variation: Dict[str, float]  # Coefficient of variation at each level
    variance_ratios: Dict[str, float]  # Variance ratios
    amplification_ratios: Dict[str, float]  # Amplification ratios
    overall_bullwhip_effect: float  # Overall bullwhip effect
    information_sharing_impact: Optional[float] = None  # Information sharing impact

@dataclass
class OrderVariabilityAnalysis:
    """Order variability analysis"""
    role: str
    orders: List[int]
    mean_order: float
    std_order: float
    coefficient_of_variation: float
    min_order: int
    max_order: int
    order_range: int
    trend_slope: Optional[float] = None

class BullwhipAnalyzer:
    """Bullwhip effect analyzer"""

    def __init__(self):
        self.logger = logging.getLogger("BullwhipAnalyzer")

    def analyze_simulation_result(self, result: 'SimulationResult') -> BullwhipMetrics:
        """Analyze bullwhip effect in simulation results

        Args:
            result: Simulation result

        Returns:
            Bullwhip effect metrics
        """
        try:
            import numpy as np
        except ImportError:
            self.logger.error("NumPy is required for bullwhip analysis")
            raise ImportError("Please install numpy: pip install numpy")

        # Extract order data
        order_data = self._extract_order_data(result.round_history)

        # Calculate variability at each level
        variability_analysis = self._analyze_order_variability(order_data)

        # Calculate bullwhip effect metrics
        cv_dict = {analysis.role: analysis.coefficient_of_variation
                  for analysis in variability_analysis}

        variance_ratios = self._calculate_variance_ratios(variability_analysis)
        amplification_ratios = self._calculate_amplification_ratios(variability_analysis)
        overall_effect = self._calculate_overall_bullwhip_effect(amplification_ratios)

        return BullwhipMetrics(
            coefficient_of_variation=cv_dict,
            variance_ratios=variance_ratios,
            amplification_ratios=amplification_ratios,
            overall_bullwhip_effect=overall_effect
        )

    def _extract_order_data(self, round_history: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Extract order data from round history"""
        order_data = {
            'customer': [],  # Customer demand
            'retailer': [],
            'wholesaler': [],
            'distributor': [],
            'manufacturer': []
        }

        for round_data in round_history:
            # Customer demand
            order_data['customer'].append(round_data['customer_demand'])

            # Agent orders at each level
            for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                if role in round_data['agents']:
                    order_placed = round_data['agents'][role]['order_placed']
                    order_data[role].append(order_placed)

        return order_data

    def _analyze_order_variability(self, order_data: Dict[str, List[int]]) -> List[OrderVariabilityAnalysis]:
        """Analyze order variability at each level"""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required for variability analysis")

        analyses = []

        for role, orders in order_data.items():
            if len(orders) < 2:
                continue

            orders_array = np.array(orders)
            mean_order = np.mean(orders_array)
            std_order = np.std(orders_array)

            # Avoid division by zero
            cv = std_order / mean_order if mean_order > 0 else 0

            # Calculate trend
            trend_slope = None
            if len(orders) > 2:
                x = np.arange(len(orders))
                trend_slope = np.polyfit(x, orders_array, 1)[0]

            analysis = OrderVariabilityAnalysis(
                role=role,
                orders=orders,
                mean_order=mean_order,
                std_order=std_order,
                coefficient_of_variation=cv,
                min_order=int(np.min(orders_array)),
                max_order=int(np.max(orders_array)),
                order_range=int(np.max(orders_array) - np.min(orders_array)),
                trend_slope=trend_slope
            )

            analyses.append(analysis)

        return analyses

    def _calculate_variance_ratios(self, analyses: List[OrderVariabilityAnalysis]) -> Dict[str, float]:
        """Calculate variance ratios"""
        variance_ratios = {}

        # Arrange by supply chain order
        supply_chain_order = ['customer', 'retailer', 'wholesaler', 'distributor', 'manufacturer']
        analysis_dict = {a.role: a for a in analyses}

        for i in range(len(supply_chain_order) - 1):
            downstream = supply_chain_order[i]
            upstream = supply_chain_order[i + 1]

            if downstream in analysis_dict and upstream in analysis_dict:
                downstream_var = analysis_dict[downstream].std_order ** 2
                upstream_var = analysis_dict[upstream].std_order ** 2

                if downstream_var > 0:
                    ratio = upstream_var / downstream_var
                    variance_ratios[f"{upstream}_vs_{downstream}"] = ratio

        return variance_ratios

    def _calculate_amplification_ratios(self, analyses: List[OrderVariabilityAnalysis]) -> Dict[str, float]:
        """Calculate amplification ratios (coefficient of variation ratios)"""
        amplification_ratios = {}

        # Arrange by supply chain order
        supply_chain_order = ['customer', 'retailer', 'wholesaler', 'distributor', 'manufacturer']
        analysis_dict = {a.role: a for a in analyses}

        for i in range(len(supply_chain_order) - 1):
            downstream = supply_chain_order[i]
            upstream = supply_chain_order[i + 1]

            if downstream in analysis_dict and upstream in analysis_dict:
                downstream_cv = analysis_dict[downstream].coefficient_of_variation
                upstream_cv = analysis_dict[upstream].coefficient_of_variation

                if downstream_cv > 0:
                    ratio = upstream_cv / downstream_cv
                    amplification_ratios[f"{upstream}_vs_{downstream}"] = ratio

        return amplification_ratios

    def _calculate_overall_bullwhip_effect(self, amplification_ratios: Dict[str, float]) -> float:
        """Calculate overall bullwhip effect"""
        if not amplification_ratios:
            return 1.0

        # Use maximum amplification ratio as overall bullwhip effect metric
        return max(amplification_ratios.values())

    def compare_scenarios(self,
                         baseline_result: 'SimulationResult',
                         comparison_result: 'SimulationResult') -> Dict[str, Any]:
        """Compare bullwhip effect between two scenarios

        Args:
            baseline_result: Baseline scenario result
            comparison_result: Comparison scenario result

        Returns:
            Comparison analysis result
        """
        baseline_metrics = self.analyze_simulation_result(baseline_result)
        comparison_metrics = self.analyze_simulation_result(comparison_result)

        # Calculate improvement degree
        improvement = {
            'overall_bullwhip_reduction': (
                baseline_metrics.overall_bullwhip_effect -
                comparison_metrics.overall_bullwhip_effect
            ) / baseline_metrics.overall_bullwhip_effect * 100,
            'cost_reduction': (
                baseline_result.total_cost - comparison_result.total_cost
            ) / baseline_result.total_cost * 100
        }

        # Improvements at each level
        level_improvements = {}
        for role in baseline_metrics.coefficient_of_variation:
            if role in comparison_metrics.coefficient_of_variation:
                baseline_cv = baseline_metrics.coefficient_of_variation[role]
                comparison_cv = comparison_metrics.coefficient_of_variation[role]

                if baseline_cv > 0:
                    reduction = (baseline_cv - comparison_cv) / baseline_cv * 100
                    level_improvements[role] = reduction

        return {
            'baseline_metrics': baseline_metrics,
            'comparison_metrics': comparison_metrics,
            'improvement': improvement,
            'level_improvements': level_improvements,
            'summary': self._generate_comparison_summary(baseline_metrics, comparison_metrics, improvement)
        }

    def _generate_comparison_summary(self,
                                   baseline: BullwhipMetrics,
                                   comparison: BullwhipMetrics,
                                   improvement: Dict[str, float]) -> str:
        """Generate comparison summary"""
        summary = f"""Bullwhip Effect Comparison Analysis:

Overall Bullwhip Effect:
- Baseline scenario: {baseline.overall_bullwhip_effect:.3f}
- Comparison scenario: {comparison.overall_bullwhip_effect:.3f}
- Improvement: {improvement['overall_bullwhip_reduction']:.1f}%

Cost Impact:
- Cost reduction: {improvement['cost_reduction']:.1f}%

Coefficient of Variation Comparison by Level:"""

        for role in baseline.coefficient_of_variation:
            if role in comparison.coefficient_of_variation:
                baseline_cv = baseline.coefficient_of_variation[role]
                comparison_cv = comparison.coefficient_of_variation[role]
                summary += f"\n- {role}: {baseline_cv:.3f} → {comparison_cv:.3f}"

        return summary

    def analyze_information_sharing_impact(self,
                                         no_sharing_result: 'SimulationResult',
                                         with_sharing_result: 'SimulationResult') -> Dict[str, Any]:
        """Analyze the impact of information sharing on bullwhip effect

        Args:
            no_sharing_result: Result without information sharing
            with_sharing_result: Result with information sharing

        Returns:
            Information sharing impact analysis
        """
        comparison = self.compare_scenarios(no_sharing_result, with_sharing_result)

        # Add information sharing specific analysis
        sharing_impact = {
            'bullwhip_reduction_by_sharing': comparison['improvement']['overall_bullwhip_reduction'],
            'cost_savings_by_sharing': comparison['improvement']['cost_reduction'],
            'most_benefited_level': max(
                comparison['level_improvements'].items(),
                key=lambda x: x[1]
            )[0] if comparison['level_improvements'] else None,
            'sharing_effectiveness': self._evaluate_sharing_effectiveness(
                comparison['improvement']['overall_bullwhip_reduction']
            )
        }

        comparison['sharing_impact'] = sharing_impact
        return comparison

    def _evaluate_sharing_effectiveness(self, reduction_percentage: float) -> str:
        """Evaluate information sharing effectiveness"""
        if reduction_percentage > 30:
            return "Very Effective"
        elif reduction_percentage > 15:
            return "Effective"
        elif reduction_percentage > 5:
            return "Slightly Effective"
        elif reduction_percentage > 0:
            return "Marginally Effective"
        else:
            return "Ineffective or Negative Impact"

    def generate_recommendations(self, metrics: BullwhipMetrics) -> List[str]:
        """Generate improvement suggestions based on bullwhip effect analysis

        Args:
            metrics: Bullwhip effect metrics

        Returns:
            List of improvement suggestions
        """
        recommendations = []

        # Based on overall bullwhip effect severity
        if metrics.overall_bullwhip_effect > 3.0:
            recommendations.append("Severe bullwhip effect. Recommend enabling information sharing mechanism")
            recommendations.append("Consider implementing Vendor Managed Inventory (VMI) strategy")
            recommendations.append("Reduce batch ordering, adopt more frequent small-batch replenishment")

        elif metrics.overall_bullwhip_effect > 2.0:
            recommendations.append("Moderate bullwhip effect. Recommend optimizing forecasting methods")
            recommendations.append("Consider implementing Collaborative Planning, Forecasting and Replenishment (CPFR)")

        elif metrics.overall_bullwhip_effect > 1.5:
            recommendations.append("Mild bullwhip effect. Recommend fine-tuning ordering strategy")

        else:
            recommendations.append("Bullwhip effect well controlled. Maintain current strategy")

        # Based on coefficient of variation at each level
        high_variability_roles = [
            role for role, cv in metrics.coefficient_of_variation.items()
            if cv > 0.5
        ]

        if high_variability_roles:
            recommendations.append(
                f"High order variability in the following roles, need focused attention: {', '.join(high_variability_roles)}"
            )

        # Based on amplification ratios
        high_amplification = [
            pair for pair, ratio in metrics.amplification_ratios.items()
            if ratio > 2.0
        ]

        if high_amplification:
            recommendations.append(
                f"Significant demand amplification exists in the following: {', '.join(high_amplification)}"
            )
            recommendations.append("Recommend implementing demand smoothing strategy at these stages")

        return recommendations
