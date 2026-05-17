"""Visualization Module

Provides chart display functionality for simulation results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Ensure Chinese font displays correctly
try:
    import matplotlib.font_manager as fm
    # Search for available Chinese fonts on the system
    font_list = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'Arial Unicode' in f.name]
    if font_list:
        plt.rcParams['font.sans-serif'] = font_list + ['DejaVu Sans', 'sans-serif']
except Exception:
    pass


class BeerGameVisualizer:
    """Beer Game Visualizer"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'retailer': '#EF4444',
            'wholesaler': '#10B981',
            'distributor': '#3B82F6',
            'manufacturer': '#F59E0B',
            'demand': '#A78BFA',
            'cost': '#F43F5E'
        }
    
    def plot_inventory_levels(self, simulation_result, save_path: Optional[str] = None) -> None:
        """Plot inventory levels chart"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get inventory history data from agent_states
        retailer_inventory = simulation_result.agent_states['retailer']['inventory_history']
        wholesaler_inventory = simulation_result.agent_states['wholesaler']['inventory_history']
        distributor_inventory = simulation_result.agent_states['distributor']['inventory_history']
        manufacturer_inventory = simulation_result.agent_states['manufacturer']['inventory_history']
        
        weeks = list(range(1, len(retailer_inventory) + 1))
        
        # Plot inventory for each role
        ax.plot(weeks, retailer_inventory,
                label='Retailer', color=self.colors['retailer'], linewidth=2)
        ax.plot(weeks, wholesaler_inventory,
                label='Wholesaler', color=self.colors['wholesaler'], linewidth=2)
        ax.plot(weeks, distributor_inventory,
                label='Distributor', color=self.colors['distributor'], linewidth=2)
        ax.plot(weeks, manufacturer_inventory,
                label='Manufacturer', color=self.colors['manufacturer'], linewidth=2)

        ax.set_xlabel('Week')
        ax.set_ylabel('Inventory Level')
        ax.set_title('Supply Chain Inventory Level Changes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_orders_and_demand(self, simulation_result, save_path: Optional[str] = None) -> None:
        """Plot orders and demand chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))

        # Extract demand data from round_history
        demand_history = [round_data.get('demand', 0) for round_data in simulation_result.round_history]
        weeks = list(range(1, len(demand_history) + 1))
        
        # Upper plot: demand and orders
        ax1.plot(weeks, demand_history,
                label='Customer Demand', color=self.colors['demand'], linewidth=3, marker='o')
        # Get order history data from agent_states
        retailer_orders = simulation_result.agent_states['retailer']['orders_history']
        wholesaler_orders = simulation_result.agent_states['wholesaler']['orders_history']
        distributor_orders = simulation_result.agent_states['distributor']['orders_history']

        ax1.plot(weeks[:-1], retailer_orders[:-1] if len(retailer_orders) > 1 else [],
                 label='Retailer', color=self.colors['retailer'], linewidth=2)
        ax1.plot(weeks[:-1], wholesaler_orders[:-1] if len(wholesaler_orders) > 1 else [],
                 label='Wholesaler', color=self.colors['wholesaler'], linewidth=2)
        ax1.plot(weeks[:-1], distributor_orders[:-1] if len(distributor_orders) > 1 else [],
                 label='Distributor', color=self.colors['distributor'], linewidth=2)

        ax1.set_xlabel('Week')
        ax1.set_ylabel('Quantity')
        ax1.set_title('Demand vs. Orders')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Lower plot: backorder status
        # Get inventory history data from agent_states
        retailer_inventory = simulation_result.agent_states['retailer']['inventory_history']
        wholesaler_inventory = simulation_result.agent_states['wholesaler']['inventory_history']
        distributor_inventory = simulation_result.agent_states['distributor']['inventory_history']
        manufacturer_inventory = simulation_result.agent_states['manufacturer']['inventory_history']
        
        # Calculate backorder amounts (negative inventory)
        retailer_backorders = [max(0, -inv) for inv in retailer_inventory]
        wholesaler_backorders = [max(0, -inv) for inv in wholesaler_inventory]
        distributor_backorders = [max(0, -inv) for inv in distributor_inventory]
        manufacturer_backorders = [max(0, -inv) for inv in manufacturer_inventory]

        ax2.plot(weeks, retailer_backorders,
                 label='Retailer', color=self.colors['retailer'], linewidth=2)
        ax2.plot(weeks, wholesaler_backorders,
                 label='Wholesaler', color=self.colors['wholesaler'], linewidth=2)
        ax2.plot(weeks, distributor_backorders,
                 label='Distributor', color=self.colors['distributor'], linewidth=2)
        ax2.plot(weeks, manufacturer_backorders,
                 label='Manufacturer', color=self.colors['manufacturer'], linewidth=2)

        ax2.set_xlabel('Week')
        ax2.set_ylabel('Backorder Quantity')
        ax2.set_title('Backorder Status')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_costs(self, simulation_result, save_path: Optional[str] = None) -> None:
        """Plot cost analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        # Cumulative costs
        # Get cost history data from agent_states
        retailer_cost_history = simulation_result.agent_states['retailer']['cost_history']
        wholesaler_cost_history = simulation_result.agent_states['wholesaler']['cost_history']
        distributor_cost_history = simulation_result.agent_states['distributor']['cost_history']
        manufacturer_cost_history = simulation_result.agent_states['manufacturer']['cost_history']
        
        weeks = list(range(1, len(retailer_cost_history) + 1))
        
        retailer_costs = np.cumsum(retailer_cost_history)
        wholesaler_costs = np.cumsum(wholesaler_cost_history)
        distributor_costs = np.cumsum(distributor_cost_history)
        manufacturer_costs = np.cumsum(manufacturer_cost_history)
        
        ax1.plot(weeks, retailer_costs, label='Retailer', color=self.colors['retailer'], linewidth=2)
        ax1.plot(weeks, wholesaler_costs, label='Wholesaler', color=self.colors['wholesaler'], linewidth=2)
        ax1.plot(weeks, distributor_costs, label='Distributor', color=self.colors['distributor'], linewidth=2)
        ax1.plot(weeks, manufacturer_costs, label='Manufacturer', color=self.colors['manufacturer'], linewidth=2)

        ax1.set_xlabel('Week')
        ax1.set_ylabel('Cumulative Cost')
        ax1.set_title('Cumulative Cost Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Total cost pie chart
        total_costs = [
            retailer_costs[-1],
            wholesaler_costs[-1],
            distributor_costs[-1],
            manufacturer_costs[-1]
        ]
        labels = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.colors['retailer'], self.colors['wholesaler'], 
                 self.colors['distributor'], self.colors['manufacturer']]
        
        ax2.pie(total_costs, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Total Cost Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shipments(self, simulation_result, save_path: Optional[str] = None) -> None:
        """Plot shipment volume chart for each participant"""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Determine complete simulation round range
        total_rounds = len(simulation_result.round_history)
        weeks = list(range(1, total_rounds + 1))
        
        # Get shipment history data from agent_states
        retailer_shipments = simulation_result.agent_states['retailer'].get('shipment_history', [])
        wholesaler_shipments = simulation_result.agent_states['wholesaler'].get('shipment_history', [])
        distributor_shipments = simulation_result.agent_states['distributor'].get('shipment_history', [])
        manufacturer_shipments = simulation_result.agent_states['manufacturer'].get('shipment_history', [])
        
        # Helper function to pad shipment data
        def pad_shipment_data(shipments, target_length):
            if len(shipments) < target_length:
                return shipments + [0] * (target_length - len(shipments))
            return shipments[:target_length]
        
        if total_rounds == 0:
            ax.text(0.5, 0.5, 'No shipment data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Shipment Volume by Participant')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        # Pad all shipment data to full round length
        retailer_shipments_padded = pad_shipment_data(retailer_shipments, total_rounds)
        wholesaler_shipments_padded = pad_shipment_data(wholesaler_shipments, total_rounds)
        distributor_shipments_padded = pad_shipment_data(distributor_shipments, total_rounds)
        manufacturer_shipments_padded = pad_shipment_data(manufacturer_shipments, total_rounds)

        # Plot shipment volume for each role using full round range
        ax.plot(weeks, retailer_shipments_padded,
                label='Retailer', color=self.colors['retailer'], linewidth=2, marker='o')
        ax.plot(weeks, wholesaler_shipments_padded,
                label='Wholesaler', color=self.colors['wholesaler'], linewidth=2, marker='s')
        ax.plot(weeks, distributor_shipments_padded,
                label='Distributor', color=self.colors['distributor'], linewidth=2, marker='^')
        ax.plot(weeks, manufacturer_shipments_padded,
                label='Manufacturer', color=self.colors['manufacturer'], linewidth=2, marker='d')

        ax.set_xlabel('Week')
        ax.set_ylabel('Shipment Volume')
        ax.set_title('Shipment Volume Changes by Participant')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value annotations (only for last few points)
        for i, (shipments, color, label) in enumerate([
            (retailer_shipments, self.colors['retailer'], 'Retailer'),
            (wholesaler_shipments, self.colors['wholesaler'], 'Wholesaler'),
            (distributor_shipments, self.colors['distributor'], 'Distributor'),
            (manufacturer_shipments, self.colors['manufacturer'], 'Manufacturer')
        ]):
            if shipments and len(shipments) > 0:
                last_week = len(shipments)
                last_value = shipments[-1]
                ax.annotate(f'{last_value}', 
                           xy=(last_week, last_value), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color=color, weight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shipments_interactive(self, simulation_result, save_path: Optional[str] = None) -> go.Figure:
        """Plot shipment volume chart for each participant (interactive)"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Retailer Shipments', 'Wholesaler Shipments', 'Distributor Shipments', 'Manufacturer Shipments')
        )
        
        # Determine complete simulation round range
        total_rounds = len(simulation_result.round_history)
        weeks = list(range(1, total_rounds + 1))
        
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        
        # Helper function to pad shipment data
        def pad_shipment_data(shipments, target_length):
            if len(shipments) < target_length:
                return shipments + [0] * (target_length - len(shipments))
            return shipments[:target_length]
        
        for i, role in enumerate(roles):
            # Get shipment history data
            shipment_history = simulation_result.agent_states[role].get('shipment_history', [])

            # Pad to full cycle length
            shipment_history_padded = pad_shipment_data(shipment_history, total_rounds)
            
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=weeks, y=shipment_history_padded,
                    mode='lines+markers',
                    name=f'{role} Shipment',
                    line=dict(color=self.colors[role], width=2),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Supply Chain Shipment Volume by Stage',
            height=600,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Shipment Volume")
        
        return fig
    
    def plot_bullwhip_effect(self, bullwhip_metrics, save_path: Optional[str] = None) -> None:
        """Plot bullwhip effect analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        # Coefficient of variation comparison
        roles = []
        cvs = []
        role_mapping = {
            'customer': 'Customer Demand',
            'retailer': 'Retailer',
            'wholesaler': 'Wholesaler',
            'distributor': 'Distributor',
            'manufacturer': 'Manufacturer'
        }
        
        for role, cv in bullwhip_metrics.coefficient_of_variation.items():
            roles.append(role_mapping.get(role, role))
            cvs.append(cv)
        
        # Dynamically set colors
        colors = []
        for role in bullwhip_metrics.coefficient_of_variation.keys():
            if role == 'customer':
                colors.append(self.colors['demand'])
            else:
                colors.append(self.colors.get(role, '#666666'))
        
        bars = ax1.bar(roles, cvs, color=colors)
        ax1.set_ylabel('Coefficient of Variation (CV)')
        ax1.set_title('Order Variability Comparison')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, cv in zip(bars, cvs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{cv:.3f}', ha='center', va='bottom')

        # Amplification ratios
        amp_pairs = list(bullwhip_metrics.amplification_ratios.keys())
        amplification_ratios = list(bullwhip_metrics.amplification_ratios.values())
        
        bars2 = ax2.bar(amp_pairs, amplification_ratios, 
                        color=[self.colors.get('retailer', '#666666')] * len(amp_pairs))
        ax2.set_ylabel('Amplification Ratio')
        ax2.set_title('Bullwhip Effect Amplification Ratio')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Add value labels
        for bar, ratio in zip(bars2, amplification_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, simulation_result, bullwhip_metrics) -> go.Figure:
        """Create interactive dashboard"""
        # Create subplots - add shipment volume chart
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Inventory Level', 'Orders & Demand', 'Cumulative Cost', 'Shipment Volume', 'Bullwhip Effect', ''),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get data from agent_states
        retailer_inventory = simulation_result.agent_states['retailer']['inventory_history']
        wholesaler_inventory = simulation_result.agent_states['wholesaler']['inventory_history']
        distributor_inventory = simulation_result.agent_states['distributor']['inventory_history']
        manufacturer_inventory = simulation_result.agent_states['manufacturer']['inventory_history']
        
        weeks = list(range(1, len(retailer_inventory) + 1))
        
        # Inventory levels
        fig.add_trace(
            go.Scatter(x=weeks, y=retailer_inventory,
                      name='Retailer Inventory', line=dict(color=self.colors['retailer'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=wholesaler_inventory,
                      name='Wholesaler Inventory', line=dict(color=self.colors['wholesaler'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=distributor_inventory,
                      name='Distributor Inventory', line=dict(color=self.colors['distributor'])),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=manufacturer_inventory,
                      name='Manufacturer Inventory', line=dict(color=self.colors['manufacturer'])),
            row=1, col=1
        )
        
        # Orders and demand
        demand_history = [round_data.get('demand', 0) for round_data in simulation_result.round_history]
        fig.add_trace(
            go.Scatter(x=weeks, y=demand_history,
                      name='Customer Demand', line=dict(color=self.colors['demand'], width=3)),
            row=1, col=2
        )
        # Get order history data
        retailer_orders = simulation_result.agent_states['retailer']['orders_history']

        fig.add_trace(
            go.Scatter(x=weeks[:-1], y=retailer_orders[:-1] if len(retailer_orders) > 1 else [],
                      name='Retailer Orders', line=dict(color=self.colors['retailer'])),
            row=1, col=2
        )
        
        # Cumulative costs
        retailer_cost_history = simulation_result.agent_states['retailer']['cost_history']
        wholesaler_cost_history = simulation_result.agent_states['wholesaler']['cost_history']
        distributor_cost_history = simulation_result.agent_states['distributor']['cost_history']
        manufacturer_cost_history = simulation_result.agent_states['manufacturer']['cost_history']
        
        retailer_costs = np.cumsum(retailer_cost_history)
        wholesaler_costs = np.cumsum(wholesaler_cost_history)
        distributor_costs = np.cumsum(distributor_cost_history)
        manufacturer_costs = np.cumsum(manufacturer_cost_history)
        
        fig.add_trace(
            go.Scatter(x=weeks, y=retailer_costs, name='Retailer Cost',
                      line=dict(color=self.colors['retailer'])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=wholesaler_costs, name='Wholesaler Cost',
                      line=dict(color=self.colors['wholesaler'])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=distributor_costs, name='Distributor Cost',
                      line=dict(color=self.colors['distributor'])),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=manufacturer_costs, name='Manufacturer Cost',
                      line=dict(color=self.colors['manufacturer'])),
            row=2, col=1
        )
        
        # Shipment volume - use full simulation period
        retailer_shipments = simulation_result.agent_states['retailer'].get('shipment_history', [])
        wholesaler_shipments = simulation_result.agent_states['wholesaler'].get('shipment_history', [])
        distributor_shipments = simulation_result.agent_states['distributor'].get('shipment_history', [])
        manufacturer_shipments = simulation_result.agent_states['manufacturer'].get('shipment_history', [])
        
        # Use the same full cycle range as other charts
        total_weeks = len(weeks)
        
        # Pad shipment data to full cycle length
        def pad_shipment_data(shipments, target_length):
            if len(shipments) < target_length:
                return shipments + [0] * (target_length - len(shipments))
            return shipments[:target_length]
        
        retailer_shipments_padded = pad_shipment_data(retailer_shipments, total_weeks)
        wholesaler_shipments_padded = pad_shipment_data(wholesaler_shipments, total_weeks)
        distributor_shipments_padded = pad_shipment_data(distributor_shipments, total_weeks)
        manufacturer_shipments_padded = pad_shipment_data(manufacturer_shipments, total_weeks)
        
        # Plot shipment data using full weeks range
        fig.add_trace(
            go.Scatter(x=weeks, y=retailer_shipments_padded,
                      name='Retailer Shipment', line=dict(color=self.colors['retailer']),
                      mode='lines+markers'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=wholesaler_shipments_padded,
                      name='Wholesaler Shipment', line=dict(color=self.colors['wholesaler']),
                      mode='lines+markers'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=distributor_shipments_padded,
                      name='Distributor Shipment', line=dict(color=self.colors['distributor']),
                      mode='lines+markers'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=weeks, y=manufacturer_shipments_padded,
                      name='Manufacturer Shipment', line=dict(color=self.colors['manufacturer']),
                      mode='lines+markers'),
            row=2, col=2
        )
        
        # Bullwhip effect
        amp_pairs = list(bullwhip_metrics.amplification_ratios.keys())
        amplifications = list(bullwhip_metrics.amplification_ratios.values())

        fig.add_trace(
            go.Bar(x=amp_pairs, y=amplifications, name='Amplification Ratio',
                  marker_color=[self.colors.get('retailer', '#666666')] * len(amp_pairs)),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Beer Game Simulation Results Dashboard",
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text="Period", row=1, col=1)
        fig.update_yaxes(title_text="Inventory Quantity", row=1, col=1)

        fig.update_xaxes(title_text="Period", row=1, col=2)
        fig.update_yaxes(title_text="Quantity", row=1, col=2)

        fig.update_xaxes(title_text="Period", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Cost", row=2, col=1)

        fig.update_xaxes(title_text="Period", row=2, col=2)
        fig.update_yaxes(title_text="Shipment Volume", row=2, col=2)

        fig.update_xaxes(title_text="Supply Chain Stage", row=3, col=1)
        fig.update_yaxes(title_text="Amplification Ratio", row=3, col=1)
        
        return fig
    
    def save_all_plots(self, simulation_result, bullwhip_metrics, output_dir: str) -> None:
        """Save all charts"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save static charts
        self.plot_inventory_levels(simulation_result,
                                 str(output_path / "inventory_levels.png"))
        self.plot_orders_and_demand(simulation_result,
                                  str(output_path / "orders_demand.png"))
        self.plot_costs(simulation_result,
                       str(output_path / "costs.png"))
        self.plot_bullwhip_effect(bullwhip_metrics,
                                str(output_path / "bullwhip_effect.png"))

        # Save interactive charts
        interactive_fig = self.create_interactive_dashboard(simulation_result, bullwhip_metrics)
        interactive_fig.write_html(str(output_path / "interactive_dashboard.html"))

        print(f"All charts saved to: {output_path}")


class StreamlitApp:
    """Streamlit Web Application"""

    def __init__(self):
        self.visualizer = BeerGameVisualizer()

    def run_app(self):
        """Run Streamlit application"""
        st.set_page_config(
            page_title="LLM Beer Game Simulation",
            page_icon="🍺",
            layout="wide"
        )

        st.title("🍺 LLM Beer Game Simulation Platform")
        st.markdown("---")

        # Sidebar configuration
        with st.sidebar:
            st.header("Simulation Configuration")

            # Basic parameters
            total_weeks = st.slider("Simulation Weeks", 20, 100, 50)
            lead_time = st.slider("Lead Time", 1, 5, 2)
            info_sharing = st.checkbox("Enable Information Sharing")

            # Demand pattern
            demand_pattern = st.selectbox(
                "Demand Pattern",
                ["Step Demand", "Random Demand", "Seasonal Demand"]
            )

            # Agent types
            st.subheader("Agent Configuration")
            retailer_type = st.selectbox("Retailer Type", ["LLM Agent", "Rule Agent"])
            wholesaler_type = st.selectbox("Wholesaler Type", ["LLM Agent", "Rule Agent"])
            distributor_type = st.selectbox("Distributor Type", ["LLM Agent", "Rule Agent"])
            manufacturer_type = st.selectbox("Manufacturer Type", ["LLM Agent", "Rule Agent"])

            # Run simulation button
            if st.button("🚀 Start Simulation", type="primary"):
                st.session_state.run_simulation = True

        # Main content area
        if hasattr(st.session_state, 'run_simulation') and st.session_state.run_simulation:
            self.show_simulation_results()
        else:
            self.show_welcome_page()
    
    def show_welcome_page(self):
        """Show welcome page"""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            ## Welcome to the LLM Beer Game Simulation Platform!

            ### 🎯 Platform Features
            - **AI-Driven**: Uses large language models as intelligent agents
            - **Bullwhip Effect Analysis**: In-depth analysis of supply chain amplification effects
            - **Multiple Configurations**: Supports different demand patterns and agent types
            - **Visualization**: Rich charts and interactive dashboard

            ### 📊 Feature Modules
            1. **Simulation Configuration**: Set simulation parameters in the left panel
            2. **Real-Time Monitoring**: Observe the status of each supply chain stage
            3. **Results Analysis**: View detailed performance metrics
            4. **Comparative Study**: Compare the effects of different configurations

            ### 🚀 Getting Started
            Please set parameters in the left configuration panel, then click the "Start Simulation" button!
            """)
    
    def show_simulation_results(self):
        """Show simulation results"""
        # Here the actual simulation engine should be called
        # For demonstration, we create mock data
        st.success("🎉 Simulation Complete!")

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-Time Monitor", "📊 Results Analysis", "🔍 Bullwhip Effect", "📋 Detailed Data"])

        with tab1:
            st.subheader("Supply Chain Real-Time Status")
            # Display real-time charts here
            st.info("Real-time monitoring will be displayed during simulation runtime")

        with tab2:
            st.subheader("Simulation Results Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Cost", "$12,345", "-5.2%")
                st.metric("Average Inventory", "15.6", "+2.1%")

            with col2:
                st.metric("Backorder Count", "3", "-1")
                st.metric("Service Level", "94.2%", "+1.8%")

        with tab3:
            st.subheader("Bullwhip Effect Analysis")
            st.info("Bullwhip effect analysis results will be displayed here")

        with tab4:
            st.subheader("Detailed Simulation Data")
            st.info("Detailed simulation data tables will be displayed here")


def create_comparison_plot(results_dict: Dict[str, Any]) -> go.Figure:
    """Create multi-scenario comparison chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Cost Comparison', 'Inventory Level Comparison', 'Backorder Status Comparison', 'Bullwhip Effect Comparison')
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, (scenario_name, result) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        
        # Total cost
        total_cost = result.total_cost

        fig.add_trace(
            go.Bar(x=[scenario_name], y=[total_cost], name=f'{scenario_name}-Cost',
                  marker_color=color, showlegend=False),
            row=1, col=1
        )
    
    fig.update_layout(height=800, title_text="Multi-Scenario Simulation Comparison")
    return fig
