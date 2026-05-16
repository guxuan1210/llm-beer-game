import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

class DecisionAnalyzer:
    """
    Decision analysis tool
    Provides detailed analysis and optimization suggestions for participants' order decisions
    """

    def __init__(self):
        self.decision_history = {}
        self.performance_metrics = {}
        self.optimization_suggestions = {}

    def analyze_decision_quality(self, agent_name: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze agent decision quality

        Args:
            agent_name: Agent name
            decision_data: Decision-related data

        Returns:
            Decision quality analysis result
        """
        if agent_name not in self.decision_history:
            self.decision_history[agent_name] = []

        # Record decision history
        self.decision_history[agent_name].append(decision_data)

        # Keep history within reasonable limits
        if len(self.decision_history[agent_name]) > 50:
            self.decision_history[agent_name] = self.decision_history[agent_name][-50:]

        # Analyze decision quality
        quality_score = self._calculate_decision_quality_score(agent_name, decision_data)
        decision_pattern = self._analyze_decision_pattern(agent_name)
        risk_assessment = self._assess_decision_risk(agent_name, decision_data)

        return {
            'agent_name': agent_name,
            'quality_score': quality_score,
            'decision_pattern': decision_pattern,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_decision_quality_score(self, agent_name: str, decision_data: Dict[str, Any]) -> float:
        """
        Calculate decision quality score

        Args:
            agent_name: Agent name
            decision_data: Decision data

        Returns:
            Quality score (0-100)
        """
        try:
            # Basic metrics
            inventory = decision_data.get('inventory', 0)
            order = decision_data.get('order', 0)
            demand = decision_data.get('demand', 0)
            backorder = decision_data.get('backorder', 0)

            # Calculate individual scores
            inventory_score = self._score_inventory_management(inventory, demand, backorder)
            order_score = self._score_order_decision(order, demand, inventory)
            cost_score = self._score_cost_efficiency(decision_data)
            stability_score = self._score_decision_stability(agent_name)

            # Weighted average
            weights = {
                'inventory': 0.3,
                'order': 0.3,
                'cost': 0.25,
                'stability': 0.15
            }

            total_score = (
                inventory_score * weights['inventory'] +
                order_score * weights['order'] +
                cost_score * weights['cost'] +
                stability_score * weights['stability']
            )

            return max(0, min(100, total_score))

        except Exception as e:
            return 50.0  # Default medium score

    def _score_inventory_management(self, inventory: int, demand: int, backorder: int) -> float:
        """
        Evaluate inventory management quality
        """
        if demand == 0:
            return 70.0

        # Ideal inventory level (1-2x demand)
        ideal_min = demand * 1.0
        ideal_max = demand * 2.0

        if ideal_min <= inventory <= ideal_max:
            score = 100.0
        elif inventory < ideal_min:
            # Penalty for insufficient inventory
            shortage_ratio = (ideal_min - inventory) / ideal_min
            score = max(20, 100 - shortage_ratio * 60)
        else:
            # Penalty for excessive inventory
            excess_ratio = (inventory - ideal_max) / ideal_max
            score = max(30, 100 - excess_ratio * 40)

        # Severe penalty for backorders
        if backorder > 0:
            backorder_penalty = min(30, backorder / demand * 50)
            score -= backorder_penalty

        return max(0, score)

    def _score_order_decision(self, order: int, demand: int, inventory: int) -> float:
        """
        Evaluate order decision quality
        """
        if demand == 0:
            return 80.0 if order == 0 else 60.0

        # Ideal order quantity should replenish to reasonable inventory level
        target_inventory = demand * 1.5
        ideal_order = max(0, target_inventory - inventory)

        if ideal_order == 0:
            score = 100.0 if order == 0 else max(50, 100 - abs(order) / demand * 30)
        else:
            order_ratio = order / ideal_order if ideal_order > 0 else 0
            if 0.8 <= order_ratio <= 1.2:
                score = 100.0
            elif order_ratio < 0.8:
                score = max(40, 100 - (0.8 - order_ratio) * 100)
            else:
                score = max(30, 100 - (order_ratio - 1.2) * 50)

        return max(0, score)

    def _score_cost_efficiency(self, decision_data: Dict[str, Any]) -> float:
        """
        Evaluate cost efficiency
        """
        holding_cost = decision_data.get('holding_cost', 0)
        shortage_cost = decision_data.get('shortage_cost', 0)
        total_cost = decision_data.get('round_cost', holding_cost + shortage_cost)

        if total_cost == 0:
            return 100.0

        # Cost structure analysis
        if shortage_cost == 0:
            # No shortage cost, mainly check if holding cost is reasonable
            if holding_cost <= 10:
                return 90.0
            elif holding_cost <= 20:
                return 80.0
            else:
                return max(50, 100 - (holding_cost - 20) * 2)
        else:
            # Has shortage cost, severe penalty
            shortage_penalty = min(40, shortage_cost * 2)
            base_score = 100 - shortage_penalty

            # Holding cost also needs consideration
            if holding_cost > 15:
                base_score -= min(20, (holding_cost - 15) * 1.5)

            return max(20, base_score)

    def _score_decision_stability(self, agent_name: str) -> float:
        """
        Evaluate decision stability
        """
        if agent_name not in self.decision_history or len(self.decision_history[agent_name]) < 3:
            return 70.0

        history = self.decision_history[agent_name][-10:]  # Last 10 decisions
        orders = [d.get('order', 0) for d in history]

        if len(orders) < 2:
            return 70.0

        # Calculate standard deviation of order changes
        order_std = np.std(orders)
        order_mean = np.mean(orders)

        if order_mean == 0:
            cv = 0
        else:
            cv = order_std / abs(order_mean)  # Coefficient of variation

        # Smaller CV means better stability
        if cv <= 0.3:
            return 100.0
        elif cv <= 0.6:
            return 85.0
        elif cv <= 1.0:
            return 70.0
        else:
            return max(40, 100 - cv * 30)

    def _analyze_decision_pattern(self, agent_name: str) -> Dict[str, Any]:
        """
        Analyze decision pattern
        """
        if agent_name not in self.decision_history or len(self.decision_history[agent_name]) < 5:
            return {
                'pattern_type': 'Insufficient Data',
                'trend': 'Unknown',
                'volatility': 'Unknown',
                'description': 'More data needed for pattern analysis'
            }

        history = self.decision_history[agent_name][-20:]  # Last 20 decisions
        orders = [d.get('order', 0) for d in history]
        inventories = [d.get('inventory', 0) for d in history]
        demands = [d.get('demand', 0) for d in history]

        # Analyze trend
        trend = self._analyze_trend(orders)

        # Analyze volatility
        volatility = self._analyze_volatility(orders)

        # Identify pattern type
        pattern_type = self._identify_pattern_type(orders, demands)

        return {
            'pattern_type': pattern_type,
            'trend': trend,
            'volatility': volatility,
            'description': self._generate_pattern_description(pattern_type, trend, volatility)
        }

    def _analyze_trend(self, values: List[float]) -> str:
        """
        Analyze value trend
        """
        if len(values) < 3:
            return 'Unknown'

        # Simple linear regression to analyze trend
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.1:
            return 'Stable'
        elif slope > 0.1:
            return 'Rising'
        else:
            return 'Falling'

    def _analyze_volatility(self, values: List[float]) -> str:
        """
        Analyze volatility
        """
        if len(values) < 2:
            return 'Unknown'

        std_dev = np.std(values)
        mean_val = np.mean(values)

        if mean_val == 0:
            cv = 0
        else:
            cv = std_dev / abs(mean_val)

        if cv <= 0.2:
            return 'Low'
        elif cv <= 0.5:
            return 'Medium'
        else:
            return 'High'

    def _identify_pattern_type(self, orders: List[float], demands: List[float]) -> str:
        """
        Identify decision pattern type
        """
        if len(orders) < 5 or len(demands) < 5:
            return 'Insufficient Data'

        # Calculate correlation between orders and demands
        if len(demands) >= len(orders):
            demand_subset = demands[-len(orders):]
        else:
            demand_subset = demands
            order_subset = orders[-len(demands):]
            orders = order_subset

        try:
            correlation = np.corrcoef(orders, demand_subset)[0, 1]

            # Determine pattern based on correlation
            if correlation > 0.7:
                return 'Reasonable Following'
            elif correlation < -0.3:
                return 'Counter-Reactive'
            else:
                return 'Random Decision'

        except:
            return 'Unrecognizable'

    def _generate_pattern_description(self, pattern_type: str, trend: str, volatility: str) -> str:
        """
        Generate pattern description
        """
        descriptions = {
            'Reasonable Following': f'The agent makes reasonably sound decisions, appropriately following demand changes. Decision trend: {trend}, volatility: {volatility}.',
            'Counter-Reactive': f'The agent decisions show negative correlation with demand changes. Decision trend: {trend}, volatility: {volatility}. Decision logic needs re-evaluation.',
            'Random Decision': f'The agent decisions lack clear patterns. Decision trend: {trend}, volatility: {volatility}. Recommend establishing a more systematic decision framework.',
            'Insufficient Data': 'Insufficient data for effective pattern analysis.',
            'Unrecognizable': 'Decision pattern is complex and cannot be described by simple models.'
        }

        return descriptions.get(pattern_type, 'Unknown decision pattern')

    def _assess_decision_risk(self, agent_name: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess decision risk
        """
        inventory = decision_data.get('inventory', 0)
        order = decision_data.get('order', 0)
        demand = decision_data.get('demand', 0)
        backorder = decision_data.get('backorder', 0)

        risks = []
        risk_level = 'Low'

        # Inventory risk
        if inventory < demand * 0.5:
            risks.append('Insufficient inventory risk: current inventory may not meet demand')
            risk_level = 'High'
        elif inventory > demand * 3:
            risks.append('Excess inventory risk: holding costs may be too high')
            if risk_level == 'Low':
                risk_level = 'Medium'

        # Order risk
        if order > demand * 2:
            risks.append('Excessive order risk: may lead to future inventory buildup')
            risk_level = 'High'
        elif order == 0 and inventory < demand and backorder > 0:
            risks.append('Under-replenishment risk: inventory not replenished in time')
            if risk_level == 'Low':
                risk_level = 'Medium'

        # Backorder risk
        if backorder > demand * 0.5:
            risks.append('Severe shortage risk: customer satisfaction may be affected')
            risk_level = 'High'

        # Decision consistency risk
        if agent_name in self.decision_history and len(self.decision_history[agent_name]) >= 3:
            recent_orders = [d.get('order', 0) for d in self.decision_history[agent_name][-3:]]
            if max(recent_orders) - min(recent_orders) > demand * 2:
                risks.append('Decision instability risk: order fluctuation too large')
                if risk_level == 'Low':
                    risk_level = 'Medium'

        return {
            'risk_level': risk_level,
            'risk_factors': risks,
            'risk_count': len(risks)
        }

    def generate_optimization_suggestions(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions

        Args:
            agent_name: Agent name

        Returns:
            List of optimization suggestions
        """
        if agent_name not in self.decision_history or len(self.decision_history[agent_name]) < 3:
            return [{
                'type': 'Data Collection',
                'priority': 'Medium',
                'suggestion': 'Need to collect more decision data to provide personalized suggestions',
                'expected_impact': 'Lay the foundation for subsequent optimization'
            }]

        suggestions = []
        recent_data = self.decision_history[agent_name][-10:]

        # Analyze recent decision quality
        avg_quality = np.mean([self._calculate_decision_quality_score(agent_name, d) for d in recent_data])

        # Provide suggestions based on quality score
        if avg_quality < 60:
            suggestions.extend(self._generate_quality_improvement_suggestions(agent_name, recent_data))
        elif avg_quality < 80:
            suggestions.extend(self._generate_moderate_improvement_suggestions(agent_name, recent_data))
        else:
            suggestions.extend(self._generate_fine_tuning_suggestions(agent_name, recent_data))

        return suggestions

    def _generate_quality_improvement_suggestions(self, agent_name: str, recent_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate quality improvement suggestions (low quality decisions)
        """
        suggestions = []

        # Analyze main issues
        avg_inventory = np.mean([d.get('inventory', 0) for d in recent_data])
        avg_demand = np.mean([d.get('demand', 1) for d in recent_data])
        avg_backorder = np.mean([d.get('backorder', 0) for d in recent_data])

        if avg_backorder > avg_demand * 0.2:
            suggestions.append({
                'type': 'Inventory Management',
                'priority': 'High',
                'suggestion': 'Increase safety stock, recommend maintaining inventory at 1.5-2x demand level',
                'expected_impact': 'Reduce shortages, improve customer satisfaction'
            })

        if avg_inventory > avg_demand * 3:
            suggestions.append({
                'type': 'Cost Control',
                'priority': 'High',
                'suggestion': 'Optimize inventory level, avoid excessive stockpiling',
                'expected_impact': 'Reduce holding costs, improve capital turnover'
            })

        orders = [d.get('order', 0) for d in recent_data]
        if np.std(orders) > np.mean(orders) * 0.8:
            suggestions.append({
                'type': 'Decision Stability',
                'priority': 'Medium',
                'suggestion': 'Use moving average or exponential smoothing to smooth order decisions',
                'expected_impact': 'Reduce bullwhip effect, improve supply chain stability'
            })

        return suggestions

    def _generate_moderate_improvement_suggestions(self, agent_name: str, recent_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate moderate improvement suggestions (medium quality decisions)
        """
        suggestions = []

        # Refined analysis
        pattern_analysis = self._analyze_decision_pattern(agent_name)

        if pattern_analysis['pattern_type'] == 'Overreaction Type':
            suggestions.append({
                'type': 'Response Adjustment',
                'priority': 'Medium',
                'suggestion': 'Reduce overreaction to short-term demand fluctuations, adopt more conservative adjustment strategy',
                'expected_impact': 'Improve decision stability, reduce unnecessary costs'
            })

        if pattern_analysis['volatility'] == 'High':
            suggestions.append({
                'type': 'Smoothing Strategy',
                'priority': 'Medium',
                'suggestion': 'Implement order smoothing strategy, e.g. set min/max order quantity limits',
                'expected_impact': 'Reduce decision volatility, improve supply chain coordination'
            })

        return suggestions

    def _generate_fine_tuning_suggestions(self, agent_name: str, recent_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate fine-tuning suggestions (high quality decisions)
        """
        suggestions = []

        suggestions.append({
            'type': 'Continuous Optimization',
            'priority': 'Low',
            'suggestion': 'Current decision quality is good, recommend continuing to monitor and fine-tune parameters',
            'expected_impact': 'Maintain excellent performance, further improve efficiency'
        })

        # Check if there is room for further optimization
        avg_cost = np.mean([d.get('round_cost', 0) for d in recent_data])
        if avg_cost > 5:
            suggestions.append({
                'type': 'Cost Optimization',
                'priority': 'Low',
                'suggestion': 'Try to further optimize inventory level to reduce total cost',
                'expected_impact': 'Reduce operating costs while maintaining service level'
            })

        return suggestions

    def get_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """
        Get agent performance summary

        Args:
            agent_name: Agent name

        Returns:
            Performance summary
        """
        if agent_name not in self.decision_history:
            return {
                'agent_name': agent_name,
                'status': 'No Data',
                'message': 'This agent has no decision history data'
            }

        history = self.decision_history[agent_name]
        recent_data = history[-10:] if len(history) >= 10 else history

        # Calculate key metrics
        quality_scores = [self._calculate_decision_quality_score(agent_name, d) for d in recent_data]
        avg_quality = np.mean(quality_scores)
        quality_trend = self._analyze_trend(quality_scores)

        # Cost analysis
        costs = [d.get('round_cost', 0) for d in recent_data]
        avg_cost = np.mean(costs)

        # Service level analysis
        backorders = [d.get('backorder', 0) for d in recent_data]
        service_level = (len([b for b in backorders if b == 0]) / len(backorders)) * 100

        return {
            'agent_name': agent_name,
            'decision_count': len(history),
            'avg_quality_score': round(avg_quality, 2),
            'quality_trend': quality_trend,
            'avg_cost_per_round': round(avg_cost, 2),
            'service_level': round(service_level, 1),
            'last_updated': datetime.now().isoformat()
        }

    def export_analysis_report(self, agent_name: str = None) -> Dict[str, Any]:
        """
        Export analysis report

        Args:
            agent_name: Specific agent name, or None to export all agents

        Returns:
            Analysis report
        """
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_type': 'decision_analysis',
            'agents': {}
        }

        agents_to_analyze = [agent_name] if agent_name else list(self.decision_history.keys())

        for agent in agents_to_analyze:
            if agent in self.decision_history:
                agent_report = {
                    'performance_summary': self.get_performance_summary(agent),
                    'decision_pattern': self._analyze_decision_pattern(agent),
                    'optimization_suggestions': self.generate_optimization_suggestions(agent)
                }

                # Add detailed analysis of latest decision
                if self.decision_history[agent]:
                    latest_decision = self.decision_history[agent][-1]
                    agent_report['latest_decision_analysis'] = self.analyze_decision_quality(agent, latest_decision)

                report['agents'][agent] = agent_report

        return report
