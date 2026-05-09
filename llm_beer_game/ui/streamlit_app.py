#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Beer Game Streamlit Web UI

Provides interactive parameter configuration and result display interface.
"""

import sys
import io

# Fix UnicodeEncodeError on Windows: force stdout to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import tempfile
import zipfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent  # This is the llm_beer_game directory
print(f"Current file: {Path(__file__).resolve()}")
print(f"Parent: {Path(__file__).resolve().parent}")
print(f"Parent.parent: {Path(__file__).resolve().parent.parent}")
sys.path.insert(0, str(project_root))  # Add llm_beer_game directory to path
print(f"Added to Python path: {project_root}")  # Debug info
print(f"Config path exists: {(project_root / 'config' / 'game_config.py').exists()}")  # Debug info

# Import local modules
# Use absolute import method

# Ensure config module can be found
config_path = project_root / 'config'
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))

# Try different import methods
try:
    from config.game_config import (
        ConfigManager,
        get_default_config,
        get_info_sharing_config,
        get_rule_based_config,
        DemandPatternType,
        LLMProviderPreset,
        BUILTIN_LLM_PRESETS,
        get_preset_by_key,
        get_all_preset_keys,
    )
    print("Successfully imported from config.game_config")
except ImportError as e1:
    print(f"Failed to import from config.game_config: {e1}")
    try:
        # Use importlib dynamic import
        import importlib.util
        config_file_path = project_root.resolve() / 'config' / 'game_config.py'
        print(f"Project root: {project_root.resolve()}")
        print(f"Looking for config file at: {config_file_path}")
        print(f"Config file exists: {config_file_path.exists()}")
        
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_file_path}")
            
        spec = importlib.util.spec_from_file_location("game_config", str(config_file_path))
        game_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(game_config)
        
        ConfigManager = game_config.ConfigManager
        get_default_config = game_config.get_default_config
        get_info_sharing_config = game_config.get_info_sharing_config
        get_rule_based_config = game_config.get_rule_based_config
        DemandPatternType = game_config.DemandPatternType
        print("Successfully imported game_config using importlib")
    except Exception as e2:
        print(f"Failed to import game_config using importlib: {e2}")
        raise e2
from agents.supply_chain_agents import create_supply_chain
from llm.llm_client import LLMClientFactory, LLMManager, MockLLMClient, OpenAIClient, AnthropicClient, OllamaClient, GPTOSSClient
from simulation.game_engine import GameEngine, DemandPattern
from analysis.bullwhip_analyzer import BullwhipAnalyzer
from visualization.visualizer import BeerGameVisualizer
from visualization.supply_chain_3d import SupplyChain3DVisualizer
from utils.helpers import setup_logging, set_random_seed


class StreamlitBeerGameApp:
    """Streamlit Beer Game App"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        # Unified color config, consistent with visualizer.py
        self.role_colors = {
            'retailer': '#FF6B6B',
            'wholesaler': '#4ECDC4',
            'distributor': '#45B7D1',
            'manufacturer': '#96CEB4'
        }
        # Inventory status color config - unified three-state color standard
        self.inventory_status_colors = {
            'positive': '#2196F3',   # Blue indicates positive inventory (sufficient)
            'zero': '#FFC107',       # Yellow indicates zero inventory (no stock)
            'negative': '#F44336'    # Red indicates negative inventory (backorder)
        }
        # Other status color config
        self.status_colors = {
            'shortage': '#F44336',   # Red for shortage (consistent with negative)
            'empty': '#FFC107',      # Yellow for no stock (consistent with zero)
            'demand': '#FECA57'      # Demand line color
        }
        self.setup_page_config()
        
    def setup_page_config(self):
        """Setup page config"""
        st.set_page_config(
            page_title="LLM Beer Game Simulation System",
            page_icon="🍺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar parameter configuration"""
        st.sidebar.title("🍺 Simulation Parameter Configuration")
        
        # Basic Settings
        st.sidebar.header("Basic Settings")
        config_type = "Default Config"
        
        # Debug Features Toggle
        with st.sidebar.expander("🔧 Debug Features", expanded=False):
            enable_prompt_debug = st.checkbox(
                "Enable Prompt Debugging",
                value=False,
                help="When enabled, can view and edit each role's prompts in real-time during simulation"
            )

            if enable_prompt_debug:
                debug_realtime = st.checkbox(
                    "Show Prompts in Real-Time",
                    value=True,
                    help="Show current round prompts in real-time during simulation"
                )
                debug_allow_edit = st.checkbox(
                    "Allow Prompt Editing",
                    value=False,
                    help="Allow prompt editing during simulation (experimental feature)"
                )
            else:
                debug_realtime = False
                debug_allow_edit = False
        
        # Simulation Parameters
        st.sidebar.header("Simulation Parameters")
        num_rounds = st.sidebar.slider("Simulation Weeks", 10, 500, 50, 5)
        
        # Generate time-based default Random Seed to ensure different runs each time
        import time
        default_seed = int(time.time()) % 10000
        random_seed = st.sidebar.number_input(
            "Random Seed", 
            0, 9999, 
            default_seed,
            help="Random Seed controls simulation randomness. Same seed produces same results; different seeds produce different results."
        )
        
        # Demand Pattern (Zoom Out Display)
        with st.sidebar.expander("📊 Demand Pattern", expanded=False):
            # Demand Category Selection
            demand_category = st.selectbox(
                "Demand Category",
                ["Basic Distributions", "Common Distributions", "Mixed Distributions", "Real Scenarios", "Unstable Demand"]
            )
            
            # Display different Demand Types based on category
            if demand_category == "Basic Distributions":
                demand_pattern = st.selectbox(
                    "Demand Type",
                    ["Random Demand", "Step Demand", "Seasonal Demand", "Constant Demand"]
                )
            elif demand_category == "Common Distributions":
                demand_pattern = st.selectbox(
                    "Demand Type",
                    ["Normal Distribution", "Poisson Distribution", "Exponential Distribution", "Triangular Distribution", "Beta Distribution", "Log-Normal Distribution"]
                )
            elif demand_category == "Mixed Distributions":
                demand_pattern = st.selectbox(
                    "Demand Type",
                    ["Bimodal Distribution", "Seasonal+Random", "Trend+Cyclic", "Multi-Stage Mix", "Markov Chain"]
                )
            elif demand_category == "Unstable Demand":
                demand_pattern = st.selectbox(
                    "Demand Type",
                    ["Autoregressive", "ARMA Demand", "Jump Diffusion", "Poisson Jump", "Regime Switching", "Volatility Clustering"]
                )
            else:  # Real Scenarios
                demand_pattern = st.selectbox(
                    "Demand Type",
                    ["Promotion-Driven", "Competition", "New Product Diffusion", "Inventory Sensitive"]
                )
            
            # Parameter Configuration
            if demand_pattern == "Random Demand":
                random_min = st.slider("Min Demand", 0, 20, 0)
                random_max = st.slider("Max Demand", 5, 50, 10)
            elif demand_pattern == "Step Demand":
                initial_demand = st.slider("Initial Demand", 0, 30, 4)
                step_demand = st.slider("Step Demand", 1, 60, 8)
                step_week = st.slider("Step Week", 5, 30, 5)
            elif demand_pattern == "Seasonal Demand":
                base_demand = st.slider("Base Demand", 10, 40, 25)
                amplitude = st.slider("Seasonal Amplitude", 5, 20, 10)
                period = st.slider("Period Length", 8, 20, 12)
            elif demand_pattern == "Constant Demand":
                initial_demand = st.slider("Constant Demand Value", 1, 20, 4)
            elif demand_pattern == "Normal Distribution":
                normal_mean = st.slider("Mean", 1.0, 30.0, 10.0, 0.5)
                normal_std = st.slider("Std Dev", 0.5, 10.0, 2.0, 0.1)
            elif demand_pattern == "Poisson Distribution":
                poisson_lambda = st.slider("Lambda Parameter", 1.0, 20.0, 8.0, 0.5)
            elif demand_pattern == "Exponential Distribution":
                exp_lambda = st.slider("Lambda Parameter", 0.1, 2.0, 0.2, 0.05)
                exp_base = st.slider("Base Value", 0, 10, 3)
            elif demand_pattern == "Triangular Distribution":
                tri_min = st.slider("Min", 0, 20, 2)
                tri_mode = st.slider("Mode", 5, 25, 10)
                tri_max = st.slider("Max", 10, 40, 18)
            elif demand_pattern == "Beta Distribution":
                beta_alpha = st.slider("Alpha", 0.5, 5.0, 2.0, 0.1)
                beta_beta = st.slider("Beta", 0.5, 5.0, 2.0, 0.1)
                beta_scale = st.slider("Scale Factor", 5, 30, 15)
            elif demand_pattern == "Log-Normal Distribution":
                lognorm_mu = st.slider("Mu Parameter", 0.0, 3.0, 2.0, 0.1)
                lognorm_sigma = st.slider("Sigma Parameter", 0.1, 1.0, 0.5, 0.05)
            elif demand_pattern == "Bimodal Distribution":
                st.subheader("First Peak")
                bimodal_mean1 = st.slider("Mean1", 1.0, 15.0, 5.0, 0.5)
                bimodal_std1 = st.slider("Std Dev1", 0.5, 5.0, 1.0, 0.1)
                st.subheader("Second Peak")
                bimodal_mean2 = st.slider("Mean2", 10.0, 30.0, 15.0, 0.5)
                bimodal_std2 = st.slider("Std Dev2", 0.5, 5.0, 2.0, 0.1)
                bimodal_weight = st.slider("First Peak Weight", 0.1, 0.9, 0.6, 0.05)
            elif demand_pattern == "Seasonal+Random":
                seasonal_base = st.slider("Base Demand", 5.0, 20.0, 10.0, 0.5)
                seasonal_amplitude = st.slider("Seasonal Amplitude", 1.0, 10.0, 3.0, 0.5)
                seasonal_period = st.slider("Period Length", 6, 24, 12)
                seasonal_noise = st.slider("Random Noise Std Dev", 0.5, 5.0, 1.5, 0.1)
            elif demand_pattern == "Trend+Cyclic":
                trend_base = st.slider("Base Demand", 5.0, 15.0, 8.0, 0.5)
                trend_slope = st.slider("Trend Slope", -0.5, 0.5, 0.1, 0.01)
                trend_amplitude = st.slider("Cycle Amplitude", 1.0, 5.0, 2.0, 0.1)
                trend_period = st.slider("Period Length", 4, 16, 8)
            elif demand_pattern == "Multi-Stage Mix":
                st.info("Multi-phase mixed demand will use preset phase configuration")
                stages_info = st.text_area(
                    "Stage Description",
                    "Stage1(1-10Weeks): Normal Distribution(Mean=8, Std Dev=1.5)\nStage2(11-20Weeks): Exponential Distribution(λ=0.15, Base=5)\nStage3(21-30Weeks): Constant Demand(Value=12)",
                    disabled=True
                )
            elif demand_pattern == "Markov Chain":
                st.info("Markov Chain demand uses preset state transition matrix")
                markov_states = st.text_input("Demand States", "3,8,15", help="Demand values separated by commas")
                markov_initial = st.selectbox("Initial State", [0, 1, 2], index=1)
            elif demand_pattern == "Promotion-Driven":
                promo_base = st.slider("Base Demand", 5.0, 15.0, 8.0, 0.5)
                promo_intensity = st.slider("Promotion Intensity Multiplier", 1.5, 5.0, 3.0, 0.1)
                promo_start = st.slider("Promotion Start Week", 5, 20, 10)
                promo_duration = st.slider("Promotion Duration", 1, 8, 3)
                promo_decay = st.slider("Post-Promotion Decay", 0.1, 0.9, 0.5, 0.05)
            elif demand_pattern == "Competition":
                comp_base = st.slider("Base Demand", 8.0, 20.0, 10.0, 0.5)
                comp_share = st.slider("Market Share", 0.2, 0.8, 0.4, 0.05)
                comp_elasticity = st.slider("Competition Elasticity", 0.5, 3.0, 1.5, 0.1)
                st.info("Competition behavior will use preset competition events")
            elif demand_pattern == "New Product Diffusion":
                diffusion_potential = st.slider("Market Potential", 500, 2000, 1000, 50)
                diffusion_innovation = st.slider("Innovation Coeff", 0.01, 0.1, 0.03, 0.005)
                diffusion_imitation = st.slider("Imitation Coeff", 0.1, 0.8, 0.38, 0.02)
            elif demand_pattern == "Inventory Sensitive":
                inventory_base = st.slider("Base Demand", 8.0, 15.0, 10.0, 0.5)
                inventory_penalty = st.slider("Stockout Penalty", 0.1, 0.5, 0.3, 0.05)
                inventory_substitution = st.slider("Substitution Rate", 0.1, 0.4, 0.2, 0.02)
            
            # Volatile Demand Pattern Parameter Configuration
            elif demand_pattern == "Autoregressive":
                st.info("📈 Autoregressive (AR): Current demand depends on historical demand values")
                ar_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                ar_coeff1 = st.slider("ARCoefficient1 (φ₁)", -1.0, 1.0, 0.7, 0.05)
                ar_coeff2 = st.slider("ARCoefficient2 (φ₂)", -1.0, 1.0, -0.2, 0.05)
                ar_noise_std = st.slider("Noise Std Dev", 1.0, 20.0, 5.0, 0.5)
                st.caption("Model: X_t = c + φ₁*X_{t-1} + φ₂*X_{t-2} + ε_t")
                
            elif demand_pattern == "ARMA Demand":
                st.info("📊 ARMA Demand: Combines autoregression and moving average")
                arma_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                arma_ar_coeff1 = st.slider("ARCoefficient1", -1.0, 1.0, 0.5, 0.05)
                arma_ar_coeff2 = st.slider("ARCoefficient2", -1.0, 1.0, 0.3, 0.05)
                arma_ma_coeff1 = st.slider("MACoefficient1", -1.0, 1.0, 0.4, 0.05)
                arma_ma_coeff2 = st.slider("MACoefficient2", -1.0, 1.0, -0.2, 0.05)
                arma_noise_std = st.slider("Noise Std Dev", 1.0, 20.0, 8.0, 0.5)
                st.caption("Model: X_t = c + Σφᵢ*X_{t-i} + Σθⱼ*ε_{t-j} + ε_t")
                
            elif demand_pattern == "Jump Diffusion":
                st.info("🚀 Jump Diffusion: Continuous Change + Sudden Jump")
                jump_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                jump_drift = st.slider("Drift Rate", -0.1, 0.1, 0.02, 0.005)
                jump_volatility = st.slider("Volatility Rate", 0.05, 0.5, 0.15, 0.01)
                jump_intensity = st.slider("Jump Intensity", 0.01, 0.5, 0.1, 0.01)
                jump_mean = st.slider("Jump Mean", -0.5, 0.5, 0.0, 0.05)
                jump_std = st.slider("Jump Std Dev", 0.1, 1.0, 0.3, 0.05)
                
            elif demand_pattern == "Poisson Jump":
                st.info("⚡ Poisson Jump: Discrete Jump Events")
                poisson_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                poisson_jump_rate = st.slider("Jump Occurrence Rate", 0.1, 0.8, 0.3, 0.05)
                st.write("Jump Size Config:")
                col1, col2 = st.columns(2)
                with col1:
                    jump_size_1 = st.number_input("Jump1", -50, 50, -10)
                    jump_size_2 = st.number_input("Jump2", -50, 50, -5)
                with col2:
                    jump_size_3 = st.number_input("Jump3", -50, 50, 5)
                    jump_size_4 = st.number_input("Jump4", -50, 50, 15)
                
            elif demand_pattern == "Regime Switching":
                st.info("🔄 Regime Switching: Switch between different market regimes")
                regime_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                st.write("Low Demand Regime:")
                col1, col2 = st.columns(2)
                with col1:
                    regime_low_mean = st.slider("Low Regime Mean", 50, 150, 80, 5)
                    regime_low_std = st.slider("Low Regime Std Dev", 1, 20, 5, 1)
                with col2:
                    regime_high_mean = st.slider("High Regime Mean", 80, 200, 120, 5)
                    regime_high_std = st.slider("High Regime Std Dev", 1, 20, 8, 1)
                st.write("Transition Probability:")
                transition_low_to_high = st.slider("Low to High Transition Probability", 0.05, 0.5, 0.15, 0.01)
                transition_high_to_low = st.slider("High to Low Transition Probability", 0.05, 0.5, 0.20, 0.01)
                
            elif demand_pattern == "Volatility Clustering":
                st.info("📈 Volatility Clustering (GARCH): Time-varying volatility")
                volatility_base_demand = st.slider("Base Demand", 50, 200, 100, 5)
                volatility_alpha = st.slider("ARCHCoefficient (α)", 0.05, 0.5, 0.15, 0.01)
                volatility_beta = st.slider("GARCHCoefficient (β)", 0.3, 0.95, 0.75, 0.01)
                volatility_omega = st.slider("Constant Term (ω)", 0.5, 10.0, 2.0, 0.1)
                st.caption("Model: σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}")
            
            # Demand Preview and Validation Feature
            st.subheader("📊 Demand Preview")
            
            # Preview Parameters
            preview_weeks = st.slider("Preview Weeks", 10, 100, 50, 5)
            preview_seed = st.number_input("Preview Random Seed", 0, 9999, 42)
            
            if st.button("🔍 Generate Demand Preview", help="Based on current config, generates a demand preview chart"):
                try:
                    # Create Demand Pattern Object for Preview
                    from simulation.game_engine import DemandPattern, GameEngine
                    from config.game_config import get_default_config
                    
                    # Build Demand Config
                    demand_config = self._build_demand_config(
                        demand_pattern, demand_category, locals()
                    )
                    
                    # Create Demand Pattern Object
                    # Need to convert UI config to demand pattern parameters
                    pattern_type = self._get_pattern_type_from_ui(demand_pattern)
                    distribution_params = self._build_distribution_params(demand_config)
                    demand_pattern_obj = DemandPattern(
                        pattern_type=pattern_type,
                        base_demand=demand_config.get('initial_demand', demand_config.get('demand', 4)),
                        step_change=(demand_config.get('step_demand', 0) - demand_config.get('initial_demand', 4)) if pattern_type == 'step' else 0,
                        step_round=demand_config.get('step_week', 5),
                        seasonal_amplitude=demand_config.get('amplitude', 0.0),
                        seasonal_period=demand_config.get('period', 12),
                        random_min=demand_config.get('min_demand', 1),
                        random_max=demand_config.get('max_demand', 10),
                        distribution_params=distribution_params,
                    )
                    
                    # Create temp GameEngine instance for demand generation
                    # Create dummy agents for preview feature
                    from agents.base_agent import BaseAgent
                    
                    temp_config = get_default_config()
                    
                    class DummyAgent(BaseAgent):
                        def __init__(self, role, config):
                            super().__init__(role, config)
                        
                        def make_decision(self, state):
                            return {"order_quantity": 0}
                    
                    dummy_agents = {
                        'retailer': DummyAgent('retailer', temp_config),
                        'wholesaler': DummyAgent('wholesaler', temp_config), 
                        'distributor': DummyAgent('distributor', temp_config),
                        'manufacturer': DummyAgent('manufacturer', temp_config)
                    }
                    temp_engine = GameEngine(
                        config=temp_config,
                        agents=dummy_agents,
                        demand_pattern=demand_pattern_obj,
                        seed=preview_seed
                    )
                    
                    # Generate Preview Data
                    preview_demands = []
                    for week in range(1, preview_weeks + 1):
                        demand = temp_engine.generate_demand(week)
                        preview_demands.append(demand)
                    
                    # Create Preview Chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, preview_weeks + 1)),
                        y=preview_demands,
                        mode='lines+markers',
                        name='Demand Preview',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig.update_layout(
                        title=f"Demand PatternPreview - {demand_pattern}",
                        xaxis_title="Weeks",
                        yaxis_title="Demand Quantity",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Display Statistics Info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Demand", f"{np.mean(preview_demands):.2f}")
                    with col2:
                        st.metric("Std Dev", f"{np.std(preview_demands):.2f}")
                    with col3:
                        st.metric("Min", f"{np.min(preview_demands):.2f}")
                    with col4:
                        st.metric("Max", f"{np.max(preview_demands):.2f}")
                    
                    # Volatile Demand Pattern - Special Statistics Metrics
                    if demand_category == "Unstable Demand":
                        st.subheader("📊 Unstable Demand Characteristic Analysis")
                        self._show_unstable_demand_analysis(demand_pattern, preview_demands)
                    
                    # Demand Validation
                    st.subheader("✅ Demand Validation")
                    
                    # Basic Validation
                    validation_results = []
                    
                    # Check if Demand values are negative
                    negative_demands = [d for d in preview_demands if d < 0]
                    if negative_demands:
                        validation_results.append(("❌", "Found Negative Demand Values", f"Found {len(negative_demands)} negative Demand values"))
                    else:
                        validation_results.append(("✅", "Demand Value Check", "All Demand Values are non-negative"))
                    
                    # Check Demand Variability
                    cv = np.std(preview_demands) / np.mean(preview_demands) if np.mean(preview_demands) > 0 else 0
                    if cv > 1.0:
                        validation_results.append(("⚠️", "Demand Variability", f"CV {cv:.2f} High; may cause Supply Chain volatility"))
                    elif cv < 0.1:
                        validation_results.append(("ℹ️", "Demand Variability", f"CV {cv:.2f} Low; Demand is relatively stable"))
                    else:
                        validation_results.append(("✅", "Demand Variability", f"CV {cv:.2f} Moderate"))
                    
                    # Check Demand Trend
                    if len(preview_demands) > 10:
                        from scipy import stats
                        slope, _, r_value, p_value, _ = stats.linregress(range(len(preview_demands)), preview_demands)
                        if abs(slope) > 0.1 and p_value < 0.05:
                            trend_direction = "Rising" if slope > 0 else "Falling"
                            validation_results.append(("ℹ️", "Demand Trend", f"Detected significant{trend_direction} trend (slope: {slope:.3f})"))
                        else:
                            validation_results.append(("✅", "Demand Trend", "No significant trend; Demand is relatively stable"))
                    
                    # Volatile Demand Pattern - Special Validation
                    if demand_category == "Unstable Demand":
                        validation_results.extend(self._validate_unstable_demand_params(demand_pattern, preview_demands))
                    
                    # Display Validation Result
                    for icon, title, message in validation_results:
                        st.write(f"{icon} **{title}**: {message}")
                    
                    # Suggestions
                    st.subheader("💡 Config Suggestions")
                    
                    avg_demand = np.mean(preview_demands)
                    if avg_demand < 5:
                        st.info("💡 Average Demand is Low，Suggestions: Adjust initial Inventory appropriately to avoid excessive Inventory")
                    elif avg_demand > 20:
                        st.info("💡 Average Demand is High，Suggestions: Increase initial Inventory or shorten Lead Time")
                    
                    if cv > 0.8:
                        st.warning("⚠️ Demand Variability is High，Suggestions: Enable Information Sharing to improve Supply Chain Coordination")
                    
                    # Volatile Demand Pattern - Special Suggestions
                    if demand_category == "Unstable Demand":
                        self._show_unstable_demand_suggestions(demand_pattern, avg_demand, cv)
                    
                except Exception as e:
                    st.error(f"Preview generation failed: {str(e)}")
                    st.info("Please check if Demand Parameter Configuration is correct")
        
        # Lead Time Config
        st.sidebar.header("⏰ Lead Time Config")
        
        # Order Lead Time (collapsible)
        with st.sidebar.expander("📦 Order Lead Time", expanded=False):
            retailer_order_lead_time = st.slider("Retailer Order Lead Time", 0, 5, 0)
            wholesaler_order_lead_time = st.slider("Wholesaler Order Lead Time", 0, 5, 0)
            distributor_order_lead_time = st.slider("Distributor Order Lead Time", 0, 5, 0)
            manufacturer_order_lead_time = st.slider("Manufacturer Order Lead Time", 0, 5, 0)
        
        # Transport & Production Lead Time (collapsible)
        with st.sidebar.expander("🚚 Transport & 🏭 Production Lead Time", expanded=False):
            retailer_transport_lead_time = st.slider("Retailer Transport Lead Time", 1, 5, 2)
            wholesaler_transport_lead_time = st.slider("Wholesaler Transport Lead Time", 1, 5, 2)
            distributor_transport_lead_time = st.slider("Distributor Transport Lead Time", 1, 5, 2)
            manufacturer_production_lead_time = st.slider("Manufacturer Production Lead Time", 1, 8, 2)
        
        # 📦 Initial Inventory& In-Transit Config
        with st.sidebar.expander("📦 Initial Inventory & In-Transit", expanded=False):
            st.caption("Set initial Inventory and In-Transit Pipeline for each Role (Per-period arrival quantity, comma-separated). If length is insufficient, auto pad with 0 to effective Lead Time length.")

            retailer_initial_inventory = st.number_input("Retailer Initial Inventory", min_value=0, value=12, step=1)
            wholesaler_initial_inventory = st.number_input("Wholesaler Initial Inventory", min_value=0, value=12, step=1)
            distributor_initial_inventory = st.number_input("Distributor Initial Inventory", min_value=0, value=12, step=1)
            manufacturer_initial_inventory = st.number_input("Manufacturer Initial Inventory", min_value=0, value=12, step=1)

            retailer_initial_in_transit_str = st.text_input("Retailer Initial In-Transit (Comma-separated)", value="0,0")
            wholesaler_initial_in_transit_str = st.text_input("Wholesaler Initial In-Transit (Comma-separated)", value="0,0")
            distributor_initial_in_transit_str = st.text_input("Distributor Initial In-Transit (Comma-separated)", value="0,0")
            manufacturer_initial_in_transit_str = st.text_input("Manufacturer Initial In-Transit (Comma-separated)", value="0,0")
        
        def _parse_in_transit(s: str):
            # Parse comma-separated in-transit string to non-negative integer list; ignore non-numeric and empty items
            try:
                items = [x.strip() for x in s.split(',')]
                parsed = []
                for x in items:
                    if not x:
                        continue
                    if x.isdigit() or (x.startswith('-') and x[1:].isdigit()):
                        v = int(x)
                        parsed.append(max(0, v))
                return parsed
            except Exception:
                return []
        
        retailer_initial_in_transit = _parse_in_transit(retailer_initial_in_transit_str)
        wholesaler_initial_in_transit = _parse_in_transit(wholesaler_initial_in_transit_str)
        distributor_initial_in_transit = _parse_in_transit(distributor_initial_in_transit_str)
        manufacturer_initial_in_transit = _parse_in_transit(manufacturer_initial_in_transit_str)
        
        # Cost Parameter
        # Cost Parameter Configuration
        st.sidebar.header("💰 Cost Parameter Configuration")
        
        # Cost Setting Mode Selection
        cost_setting_mode = st.sidebar.radio(
            "Setting Mode",
            ["Unified Settings", "Independent Settings"],
            index=0,
            help="Unified Settings: All participants use same cost parameters; Independent Settings: Each participant can set different parameters"
        )
        
        # Initialize Cost Config dictionary
        if 'cost_configs' not in st.session_state:
            st.session_state.cost_configs = {
                'unified': {
                    'holding_cost': 0.5,
                    'shortage_cost': 1.0
                },
                'individual': {
                    'retailer': {'holding_cost': 0.5, 'shortage_cost': 1.0},
                    'wholesaler': {'holding_cost': 0.5, 'shortage_cost': 1.0},
                    'distributor': {'holding_cost': 0.5, 'shortage_cost': 1.0},
                    'manufacturer': {'holding_cost': 0.5, 'shortage_cost': 1.0}
                }
            }
        
        # Define Participants Mapping
        participants = {
            'retailer': '🏪 Retailer',
            'wholesaler': '🏬 Wholesaler', 
            'distributor': '🚚 Distributor',
            'manufacturer': '🏭 Manufacturer'
        }
        
        if cost_setting_mode == "Unified Settings":
            st.sidebar.subheader("📊 Unified Cost Parameters")
            
            unified_holding = st.sidebar.slider(
                "Inventory Holding Cost", 
                0.0, 2.0, 
                st.session_state.cost_configs['unified']['holding_cost'], 
                0.1,
                help="Holding cost per unit of Inventory per week"
            )
            unified_shortage = st.sidebar.slider(
                "Stockout Penalty Cost", 
                0.0, 10.0, 
                st.session_state.cost_configs['unified']['shortage_cost'], 
                0.1,
                help="Penalty cost per unit stockout per week"
            )
            # Update unified Config
            st.session_state.cost_configs['unified'] = {
                'holding_cost': unified_holding,
                'shortage_cost': unified_shortage
            }

            # Apply unified Config to all Participants
            for participant in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                st.session_state.cost_configs['individual'][participant] = {
                    'holding_cost': unified_holding,
                    'shortage_cost': unified_shortage
                }
            
            # Display Application Info
            st.sidebar.info("✅ Unified Parameters applied to all Participants")
            
        else:  # Independent Settings
            st.sidebar.subheader("🎯 Independent Cost Parameters")
            
            # Set independent parameters for each participant
            for key, name in participants.items():
                with st.sidebar.expander(f"{name} Cost Parameter", expanded=False):
                    holding = st.slider(
                        f"{name} - Inventory Holding Cost",
                        0.0, 2.0,
                        st.session_state.cost_configs['individual'][key]['holding_cost'],
                        0.1,
                        key=f"{key}_holding"
                    )
                    shortage = st.slider(
                        f"{name} - Stockout Penalty Cost",
                        0.0, 10.0,
                        st.session_state.cost_configs['individual'][key]['shortage_cost'],
                        0.1,
                        key=f"{key}_shortage"
                    )
                    # Update individual Config
                    st.session_state.cost_configs['individual'][key] = {
                        'holding_cost': holding,
                        'shortage_cost': shortage
                    }
        
        # Get current Cost Config
        current_costs = st.session_state.cost_configs['individual']
        
        # Display Current Config Summary
        with st.sidebar.expander("📋 Current Cost Config Summary", expanded=False):
            for key, name in participants.items():
                st.write(f"**{name}**")
                st.write(f"- Holding: {current_costs[key]['holding_cost']:.1f}")
                st.write(f"- Stockout: {current_costs[key]['shortage_cost']:.1f}")
                st.write("---")
        
        # LLM Config
        st.sidebar.header("LLM Config")

        # --- Load presets (built-in + custom) ---
        config_mgr = ConfigManager()
        all_presets = config_mgr.get_all_presets()
        # Build preset options: key -> display name
        preset_keys = ["__manual__"] + sorted(all_presets.keys())
        preset_display = lambda k: "Manual Config" if k == "__manual__" else f"{all_presets[k].provider_label}: {all_presets[k].name}"

        selected_preset_key = st.sidebar.selectbox(
            "Load Preset",
            preset_keys,
            format_func=preset_display,
            help="Select a preset to auto-fill LLM settings"
        )

        # Auto-fill from preset
        preset_provider = None
        preset_model = None
        preset_base_url = None
        preset_temp = 0.7
        preset_max_tokens = 150
        preset_timeout = 30
        preset_is_thinking = False
        if selected_preset_key != "__manual__":
            preset = all_presets[selected_preset_key]
            preset_provider = preset.provider_label
            preset_model = preset.model
            preset_base_url = preset.base_url
            preset_temp = preset.default_temperature
            preset_max_tokens = preset.default_max_tokens
            preset_timeout = preset.default_timeout
            preset_is_thinking = getattr(preset, 'is_thinking_model', False)
            st.sidebar.info(f"Preset: {preset.name} — {preset.description}")

        # --- Provider selection ---
        all_providers = ["Mock LLM", "OpenAI", "Anthropic", "Ollama", "GPT-OSS",
                         "DeepSeek", "Zhipu GLM", "Moonshot Kimi", "Qwen",
                         "OpenRouter", "Groq", "Together AI", "Custom OpenAI-Compatible"]

        # Map provider label -> internal provider key
        provider_key_map = {
            "Mock LLM": "mock", "OpenAI": "openai", "Anthropic": "anthropic",
            "Ollama": "ollama", "GPT-OSS": "gpt_oss",
            "DeepSeek": "deepseek", "Zhipu GLM": "zhipu",
            "Moonshot Kimi": "moonshot", "Qwen": "qwen",
            "OpenRouter": "openrouter", "Groq": "groq",
            "Together AI": "together", "Custom OpenAI-Compatible": "openai_compatible",
        }

        default_provider_index = 3  # Default: Ollama
        if preset_provider:
            for i, label in enumerate(all_providers):
                if label == preset_provider:
                    default_provider_index = i
                    break

        llm_provider = st.sidebar.selectbox(
            "LLM Provider",
            all_providers,
            index=default_provider_index
        )

        api_key = None
        model_name = preset_model
        base_url = preset_base_url
        temperature_val = preset_temp
        max_tokens_val = preset_max_tokens
        timeout_val = preset_timeout

        # Default base URLs for each provider
        _provider_default_base_urls = {
            "OpenAI": "https://api.openai.com/v1",
            "Anthropic": "",
            "Ollama": "http://localhost:11434",
            "DeepSeek": "https://api.deepseek.com",
            "Zhipu GLM": "https://open.bigmodel.cn/api/paas/v4",
            "Moonshot Kimi": "https://api.moonshot.cn/v1",
            "Qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "OpenRouter": "https://openrouter.ai/api/v1",
            "Groq": "https://api.groq.com/openai/v1",
            "Together AI": "https://api.together.xyz/v1",
            "Custom OpenAI-Compatible": "http://localhost:8080/v1",
            "GPT-OSS": "http://localhost:11434",
        }

        # Auto-update base_url when provider changes
        if "prev_llm_provider" not in st.session_state:
            st.session_state.prev_llm_provider = llm_provider
        if st.session_state.prev_llm_provider != llm_provider:
            base_url = _provider_default_base_urls.get(llm_provider, "")
            st.session_state.prev_llm_provider = llm_provider

        # Unified Base URL input (shown for all providers)
        if not base_url:
            base_url = _provider_default_base_urls.get(llm_provider, "")
        base_url = st.sidebar.text_input(
            "Base URL",
            value=base_url,
            help=f"API endpoint for {llm_provider}"
        )

        # Provider-specific fields
        if llm_provider in ("OpenAI", "DeepSeek", "Zhipu GLM", "Moonshot Kimi", "Qwen",
                            "OpenRouter", "Groq", "Together AI", "Custom OpenAI-Compatible"):
            api_key = st.sidebar.text_input("API Key", type="password",
                                            help=f"Enter your {llm_provider} API key")

        if llm_provider == "OpenAI":
            openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
            if model_name not in openai_models:
                model_name = "gpt-4o"
            model_name = st.sidebar.selectbox("Model", openai_models,
                                              index=openai_models.index(model_name) if model_name in openai_models else 0)
        elif llm_provider == "Anthropic":
            anthropic_models = ["claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-haiku-4-20250501",
                                "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            if model_name not in anthropic_models:
                model_name = "claude-sonnet-4-20250514"
            model_name = st.sidebar.selectbox("Model", anthropic_models,
                                              index=anthropic_models.index(model_name) if model_name in anthropic_models else 1)
            api_key = st.sidebar.text_input("Anthropic API Key", type="password")
        elif llm_provider == "DeepSeek":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "deepseek-chat",
                                               help="deepseek-chat (V3) or deepseek-reasoner (R1)")
        elif llm_provider == "Zhipu GLM":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "glm-4-flash",
                                               help="glm-4, glm-4-flash, glm-4-plus, etc.")
        elif llm_provider == "Moonshot Kimi":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "moonshot-v1-8k")
        elif llm_provider == "Qwen":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "qwen-max",
                                               help="qwen-max, qwen-plus, qwen-turbo, etc.")
        elif llm_provider == "OpenRouter":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "openai/gpt-4o",
                                               help="Full model string e.g. openai/gpt-4o, anthropic/claude-sonnet-4")
        elif llm_provider == "Groq":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "llama3-70b-8192")
        elif llm_provider == "Together AI":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "meta-llama/Llama-3-70b-chat-hf")
        elif llm_provider == "Custom OpenAI-Compatible":
            model_name = st.sidebar.text_input("Model", value=model_name if model_name else "",
                                               help="Enter model name for your custom endpoint")
        elif llm_provider == "Ollama":
            # Dynamically fetch available Model list
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent
                llm_beer_game_root = Path(__file__).resolve().parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                if str(llm_beer_game_root) not in sys.path:
                    sys.path.insert(0, str(llm_beer_game_root))
                from llm_beer_game.utils.ollama_utils import get_models_with_fallback, check_ollama_service, get_model_display_info

                service_available = check_ollama_service(base_url)
                if service_available:
                    model_info = get_model_display_info(base_url)
                    if model_info:
                        display_options = [info["display_name"] for info in model_info]
                        model_names_list = [info["name"] for info in model_info]
                        default_index = 0
                        if model_name:
                            for i, name in enumerate(model_names_list):
                                if model_name.lower() in name.lower():
                                    default_index = i
                                    break
                        else:
                            for i, name in enumerate(model_names_list):
                                if "gemma3:27b" in name.lower():
                                    default_index = i
                                    break
                        selected_index = st.sidebar.selectbox(
                            "Select Model",
                            range(len(display_options)),
                            index=default_index,
                            format_func=lambda x: display_options[x],
                            help="Select installed Ollama model"
                        )
                        model_name = model_names_list[selected_index]
                        selected_info = model_info[selected_index]
                        st.sidebar.info(f"Selected: {selected_info['name']} | Size: {selected_info['size_formatted']}")
                    else:
                        st.sidebar.warning("No installed models found")
                        model_name = st.sidebar.text_input("Manual Model Name", model_name if model_name else "gemma3:27b")
                else:
                    st.sidebar.error("Cannot connect to Ollama service")
                    fallback_models = get_models_with_fallback(base_url)
                    default_fb = 0
                    if model_name:
                        for i, m in enumerate(fallback_models):
                            if model_name.lower() in m.lower():
                                default_fb = i
                                break
                    else:
                        for i, m in enumerate(fallback_models):
                            if "gemma3:27b" in m.lower():
                                default_fb = i
                                break
                    model_name = st.sidebar.selectbox(
                        "Select Recommended Model",
                        fallback_models,
                        index=default_fb,
                        help="Ollama service unavailable; showing recommended model list"
                    )
            except Exception as e:
                st.sidebar.error(f"Failed to get model list: {str(e)}")
                model_name = st.sidebar.text_input("Manual Model Name", model_name if model_name else "gemma3:27b")
        elif llm_provider == "GPT-OSS":
            model_name = st.sidebar.text_input("Model Name", model_name if model_name else "gpt-oss:20b")
            if st.sidebar.button("Check GPT-OSS Service Status"):
                try:
                    from llm.llm_client import GPTOSSClient
                    client = GPTOSSClient(model_name=model_name, base_url=base_url)
                    if client.is_available():
                        st.sidebar.success("GPT-OSS service running normally")
                    else:
                        st.sidebar.error("GPT-OSS service unavailable")
                except Exception as e:
                    st.sidebar.error(f"Service check failed: {str(e)}")

        # --- Advanced settings ---
        with st.sidebar.expander("Advanced LLM Settings", expanded=False):
            temperature_val = st.slider("Temperature", 0.0, 2.0, temperature_val, 0.05)
            max_tokens_val = st.slider("Max Tokens", 10, 500, max_tokens_val, 10)
            timeout_val = st.slider("Timeout (s)", 10, 300, timeout_val, 5)

        # --- Save custom preset ---
        with st.sidebar.expander("Save Custom Preset", expanded=False):
            custom_preset_name = st.text_input("Preset Name", placeholder="e.g. My DeepSeek Config")
            custom_preset_key = st.text_input("Preset Key", placeholder="e.g. my_deepseek_v3",
                                              help="Short unique key, no spaces")
            if st.button("Save Current as Custom Preset"):
                if custom_preset_name and custom_preset_key:
                    internal_provider = provider_key_map.get(llm_provider, "openai")
                    new_preset = LLMProviderPreset(
                        name=custom_preset_name,
                        provider=internal_provider,
                        model=model_name or "",
                        base_url=base_url,
                        description=f"Custom {llm_provider} config",
                        requires_api_key=bool(api_key is not None),
                        default_temperature=temperature_val,
                        default_max_tokens=max_tokens_val,
                        default_timeout=timeout_val,
                        provider_label=llm_provider,
                    )
                    config_mgr.add_custom_preset(custom_preset_key, new_preset)
                    st.sidebar.success(f"Preset '{custom_preset_name}' saved! Reload to see it.")
                else:
                    st.sidebar.warning("Please enter both a name and key for the preset.")

            # List and manage custom presets
            custom_presets = config_mgr.load_custom_presets()
            if custom_presets:
                st.markdown("**Your Custom Presets:**")
                for ck, cp in custom_presets.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"{cp.provider_label}: {cp.name} ({cp.model})")
                    with col2:
                        if st.button("Delete", key=f"del_{ck}"):
                            config_mgr.remove_custom_preset(ck)
                            st.rerun()
        
        # Information Sharing
        st.sidebar.header("Information Sharing")
        enable_info_sharing = st.sidebar.checkbox("Enable Information Sharing", False)
        if enable_info_sharing:
            # Use columns to display side by sidecheckbox and info buttons
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                share_inventory = st.sidebar.checkbox("Share Inventory Info", True)
            with col2:
                if st.sidebar.button("ℹ️", key="info_inventory"):
                    st.sidebar.info("📘 **Share Inventory Info**: All Supply Chain Participants can see each other's Current Inventory levels. Helps understand upstream and downstream inventory status. Avoid excessive ordering.")
            
            col3, col4 = st.sidebar.columns([4, 1])
            with col3:
                share_demand = st.sidebar.checkbox("Share Demand Info", True)
            with col4:
                if st.sidebar.button("ℹ️", key="info_demand"):
                    st.sidebar.info("📘 **Share Demand Info**: All Participants can access Retailer's terminal Customer Demand history. Let every stage see real market demand. Reduce information distortion.")
                
            col5, col6 = st.sidebar.columns([4, 1])
            with col5:
                share_orders = st.sidebar.checkbox("Share Order Info", False)
            with col6:
                if st.sidebar.button("ℹ️", key="info_orders"):
                    st.sidebar.info("📘 **Share Order Info**: Participants can see other stages' historical Order Data, helping understand Order patterns and trends, and optimize own decisions.")
            
            # Add detailed explanation expander
            with st.sidebar.expander("📖 Detailed Strategy Description", expanded=False):
                st.markdown("""
                ### Information Sharing Strategy Details
                
                When Information Sharing is enabled, you can select from the following three strategies:
                
                #### 1. 📦 Share Inventory Info
                - **Content**: All Supply Chain Participants can see each other's Current Inventory levels
                - **Purpose**: Helps participants understand upstream/downstream inventory status to avoid excessive ordering
                - **Effect**: Reduces Bullwhip Effect, improves Supply Chain transparency
                
                #### 2. 📈 Share Demand Info
                - **Content**: All Participants can access Retailer's terminal Customer Demand history
                - **Purpose**: Let every stage see real market demand to reduce information distortion
                - **Effect**: Significantly reduces Bullwhip Effect, improves Demand Forecast Accuracy
                
                #### 3. 📋 Share Order Info
                - **Content**: Participants can see other stages' historical Order Data
                - **Purpose**: Helps understand Order patterns and trends to optimize own decisions
                - **Effect**: Enhances Supply Chain Coordination, reduces Order Fluctuation
                
                ### 💡 Recommended Combinations
                - **Base Combination**: Share Inventory + Share Demand (default recommended)
                - **Full Transparency**: All Three Strategies Enabled
                - **Lightweight Mode**: Only Share Demand Info
                """)
        

        # Order Quantity Limit Config
        with st.sidebar.expander("📊 Order Quantity Limit", expanded=False):
            enable_order_limits = st.checkbox(
                "Enable Min/Max Order Quantity Limit",
                value=True,
                help="Set min and max Order Quantity range limits for each Role"
            )

            # Discrete Point Select Config
            enable_discrete_points = st.checkbox(
                "Enable Discrete Point Select",
                value=False,
                help="Set Discrete Order Quantity points for each Role; Agents can only select from these points"
            )

            if enable_discrete_points:
                st.info("Please enter comma-separated integers as discrete order points for each role")

            if enable_order_limits and enable_discrete_points:
                st.warning("Note: When both limits are enabled, Discrete Point Select takes priority; range limits serve as Validation boundaries")


            # Adaptive Order Quantity Limit Config
            enable_adaptive_limits = st.checkbox(
                "Enable Adaptive Order Quantity Limit",
                value=False,
                help="Dynamically adjust Order Quantity Limits based on Retailer terminal Demand"
            )

            adaptive_method = "ratio"
            adaptive_base_min = 3
            adaptive_base_max = 8
        
            if enable_adaptive_limits:
                st.subheader("Adaptive Limit Config")
                adaptive_method = st.selectbox(
                    "Adaptive Method",
                    options=["ratio", "window", "forecast"],
                    format_func=lambda x: {
                        "ratio": "Demand Ratio Method",
                        "window": "Sliding Window Method",
                        "forecast": "Forecast Model Method"
                    }.get(x, x),
                    help="Select Adaptive Limit Calculate Method"
                )

                adaptive_base_min = st.number_input(
                    "Base Min Order Quantity",
                    min_value=1,
                    max_value=20,
                    value=3,
                    key="adaptive_base_min"
                )

                adaptive_base_max = st.number_input(
                    "Base Max Order Quantity",
                    min_value=adaptive_base_min,
                    max_value=500,
                    value=max(15, adaptive_base_min + 5),
                    key="adaptive_base_max"
                )

                # Based on selected method, display different parameters
                if adaptive_method == "ratio":
                    min_ratio = st.slider(
                        "Min Ratio",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="Min Order to Demand Ratio"
                    )
                    max_ratio = st.slider(
                        "Max Ratio",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Max Order to Demand Ratio"
                    )

                elif adaptive_method == "window":
                    window_size = st.slider(
                        "Window Size",
                        min_value=2,
                        max_value=10,
                        value=5,
                        step=1,
                        help="Sliding Window Size"
                    )
                    alpha = st.slider(
                        "Min Limit Coefficient",
                        min_value=0.5,
                        max_value=2.0,
                        value=1.0,
                        step=0.1,
                        help="Min Limit Std Dev Coefficient"
                    )
                    beta = st.slider(
                        "Max Limit Coefficient",
                        min_value=1.0,
                        max_value=3.0,
                        value=2.0,
                        step=0.1,
                        help="Max Limit Std Dev Coefficient"
                    )

                elif adaptive_method == "forecast":
                    forecast_horizon = st.slider(
                        "Forecast Periods",
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        help="Forecast future periods"
                    )
                    forecast_weight = st.slider(
                        "Forecast Weight",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        help="Forecast Value Weight"
                    )

                adaptation_rate = st.slider(
                    "Adaptation Rate",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Limit Adjust Rate(0-1)"
                )
        
            if enable_order_limits or enable_discrete_points:
                st.subheader("Retailer Order Quantity Limit")
                if enable_discrete_points:
                    retailer_discrete_points = st.text_input(
                        "Discrete Order Point",
                        value="4,8,12,16,20",
                        key="retailer_discrete",
                        help="Input comma-separated integers as optional Order points"
                    )
                if enable_order_limits and not enable_discrete_points:
                    retailer_min_order = st.number_input(
                        "Min Order Quantity",
                        min_value=0,
                        max_value=1000,
                        value=3,
                        key="retailer_min"
                    )
                    retailer_max_order = st.number_input(
                        "Max Order Quantity",
                        min_value=retailer_min_order,
                        max_value=1000,
                        value=max(10, retailer_min_order),
                        key="retailer_max"
                    )

                st.subheader("Wholesaler Order Quantity Limit")
                if enable_discrete_points:
                    wholesaler_discrete_points = st.text_input(
                        "Discrete Order Point",
                        value="4,8,12,16,20",
                        key="wholesaler_discrete",
                        help="Input comma-separated integers as optional Order points"
                    )
                if enable_order_limits and not enable_discrete_points:
                    wholesaler_min_order = st.number_input(
                        "Min Order Quantity",
                        min_value=0,
                        max_value=1000,
                        value=3,
                        key="wholesaler_min"
                    )
                    wholesaler_max_order = st.number_input(
                        "Max Order Quantity",
                        min_value=wholesaler_min_order,
                        max_value=1000,
                        value=max(10, wholesaler_min_order),
                        key="wholesaler_max"
                    )

                st.subheader("Distributor Order Quantity Limit")
                if enable_discrete_points:
                    distributor_discrete_points = st.text_input(
                        "Discrete Order Point",
                        value="4,8,12,16,20",
                        key="distributor_discrete",
                        help="Input comma-separated integers as optional Order points"
                    )
                if enable_order_limits and not enable_discrete_points:
                    distributor_min_order = st.number_input(
                        "Min Order Quantity",
                        min_value=0,
                        max_value=1000,
                        value=3,
                        key="distributor_min"
                    )
                    distributor_max_order = st.number_input(
                        "Max Order Quantity",
                        min_value=distributor_min_order,
                        max_value=1000,
                        value=max(10, distributor_min_order),
                        key="distributor_max"
                    )

                st.subheader("Manufacturer Production Quantity Limit")
                if enable_discrete_points:
                    manufacturer_discrete_points = st.text_input(
                        "Discrete Production Point",
                        value="4,8,12,16,20",
                        key="manufacturer_discrete",
                        help="Input comma-separated integers as optional Production points"
                    )
                if enable_order_limits and not enable_discrete_points:
                    manufacturer_min_order = st.number_input(
                        "Min Production Quantity",
                        min_value=0,
                        max_value=1000,
                        value=3,
                        key="manufacturer_min"
                    )
                    manufacturer_max_order = st.number_input(
                        "Max Production Quantity",
                        min_value=manufacturer_min_order,
                        max_value=1000,
                        value=max(10, manufacturer_min_order),
                        key="manufacturer_max"
                    )

                if not enable_discrete_points:
                    st.success(
                        "✅ Order Quantity Limit already enabled:\n"
                        f"• Retailer: [{retailer_min_order}, {retailer_max_order}]\n"
                        f"• Wholesaler: [{wholesaler_min_order}, {wholesaler_max_order}]\n"
                        f"• Distributor: [{distributor_min_order}, {distributor_max_order}]\n"
                        f"• Manufacturer: [{manufacturer_min_order}, {manufacturer_max_order}]\n"
                        "• Orders exceeding the limit will be automatically adjusted"
                    )
                else:
                    # Parse Discrete Point string into integer list, filtering out 0 and negative numbers
                    retailer_points = [int(x.strip()) for x in retailer_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
                    wholesaler_points = [int(x.strip()) for x in wholesaler_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
                    distributor_points = [int(x.strip()) for x in distributor_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
                    manufacturer_points = [int(x.strip()) for x in manufacturer_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]

                    st.success(
                        "✅ Discrete Order Point already enabled:\n"
                        f"• Retailer: {retailer_points}\n"
                        f"• Wholesaler: {wholesaler_points}\n"
                        f"• Distributor: {distributor_points}\n"
                        f"• Manufacturer: {manufacturer_points}\n"
                        "• Orders will be adjusted to the nearest Discrete Point"
                    )
            else:
                st.warning(
                    "⚠️ Order Quantity Limit Not Enabled\n"
                    "• Agents can place any quantity of orders\n"
                    "• May lead to extreme Order Fluctuation\n"
                    "• Suggest enabling limits to simulate realistic constraints"
                )
        
        # Build Config Dictionary
        config = {
            "config_type": config_type,
            "num_rounds": num_rounds,
            "random_seed": random_seed,
            "demand_pattern": demand_pattern,
            "llm_provider": llm_provider,
            "api_key": api_key,
            "model_name": model_name,
            "base_url": base_url,
            "llm_max_tokens": max_tokens_val,
            "llm_is_thinking_model": preset_is_thinking,
            "llm_timeout": timeout_val,
            "enable_info_sharing": enable_info_sharing,
            # Lead Time Parameters
            "retailer_order_lead_time": retailer_order_lead_time,
            "wholesaler_order_lead_time": wholesaler_order_lead_time,
            "distributor_order_lead_time": distributor_order_lead_time,
            "manufacturer_order_lead_time": manufacturer_order_lead_time,
            "retailer_transport_lead_time": retailer_transport_lead_time,
            "wholesaler_transport_lead_time": wholesaler_transport_lead_time,
            "distributor_transport_lead_time": distributor_transport_lead_time,
            "manufacturer_production_lead_time": manufacturer_production_lead_time,
            # Order Quantity Limit Config
            "enable_order_limits": enable_order_limits,
            # Discrete Point Select Config
            "enable_discrete_points": enable_discrete_points,
            # Adaptive Order Quantity Limit Config
            "enable_adaptive_limits": enable_adaptive_limits,
            # Cost Config
            "cost_configs": st.session_state.cost_configs,
            # Debug Features Config
            "enable_prompt_debug": enable_prompt_debug,
            "debug_realtime": debug_realtime,
            "debug_allow_edit": debug_allow_edit,
            # Initial Inventory& In-Transit Config
            "retailer_initial_inventory": retailer_initial_inventory,
            "wholesaler_initial_inventory": wholesaler_initial_inventory,
            "distributor_initial_inventory": distributor_initial_inventory,
            "manufacturer_initial_inventory": manufacturer_initial_inventory,
            "retailer_initial_in_transit": retailer_initial_in_transit,
            "wholesaler_initial_in_transit": wholesaler_initial_in_transit,
            "distributor_initial_in_transit": distributor_initial_in_transit,
            "manufacturer_initial_in_transit": manufacturer_initial_in_transit,
            "enable_demand_forecasting": st.session_state.get("enable_demand_forecasting", False)
        }
        
        # Add Order Quantity Limit Parameters
        if enable_order_limits and not enable_discrete_points:
            config.update({
                "retailer_min_order": retailer_min_order,
                "retailer_max_order": retailer_max_order,
                "wholesaler_min_order": wholesaler_min_order,
                "wholesaler_max_order": wholesaler_max_order,
                "distributor_min_order": distributor_min_order,
                "distributor_max_order": distributor_max_order,
                "manufacturer_min_order": manufacturer_min_order,
                "manufacturer_max_order": manufacturer_max_order
            })
        
        # Add Discrete Point Select Parameters (independent of Order Quantity Limit)
        if enable_discrete_points:
            # Parse Discrete Point string into integer list, filtering out 0 and negative numbers
            retailer_points = [int(x.strip()) for x in retailer_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
            wholesaler_points = [int(x.strip()) for x in wholesaler_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
            distributor_points = [int(x.strip()) for x in distributor_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
            manufacturer_points = [int(x.strip()) for x in manufacturer_discrete_points.split(',') if x.strip().isdigit() and int(x.strip()) > 0]
            
            config.update({
                "retailer_discrete_points": retailer_points,
                "wholesaler_discrete_points": wholesaler_points,
                "distributor_discrete_points": distributor_points,
                "manufacturer_discrete_points": manufacturer_points
            })
        
        # Add Adaptive Order Quantity Limit Parameters
        if enable_adaptive_limits:
            config.update({
                "adaptive_method": adaptive_method,
                "adaptive_base_min": adaptive_base_min,
                "adaptive_base_max": adaptive_base_max,
                "adaptation_rate": adaptation_rate
            })
            
            # Based on selected method, add specific parameters
            if adaptive_method == "ratio":
                config.update({
                    "min_ratio": min_ratio,
                    "max_ratio": max_ratio
                })
            elif adaptive_method == "window":
                config.update({
                    "window_size": window_size,
                    "alpha": alpha,
                    "beta": beta
                })
            elif adaptive_method == "forecast":
                config.update({
                    "forecast_horizon": forecast_horizon,
                    "forecast_weight": forecast_weight
                })
        
        # Add Demand Parameters
        if demand_pattern == "Random Demand":
            config.update({"random_min": random_min, "random_max": random_max})
        elif demand_pattern == "Step Demand":
            config.update({
                "initial_demand": initial_demand,
                "step_demand": step_demand,
                "step_week": step_week
            })
        elif demand_pattern == "Seasonal Demand":
            config.update({
                "base_demand": base_demand,
                "amplitude": amplitude,
                "period": period
            })
        elif demand_pattern == "Constant Demand":
            config.update({"initial_demand": initial_demand})
        elif demand_pattern == "Normal Distribution":
            config.update({"normal_mean": normal_mean, "normal_std": normal_std})
        elif demand_pattern == "Poisson Distribution":
            config.update({"poisson_lambda": poisson_lambda})
        elif demand_pattern == "Exponential Distribution":
            config.update({"exp_lambda": exp_lambda, "exp_base": exp_base})
        elif demand_pattern == "Triangular Distribution":
            config.update({"tri_min": tri_min, "tri_mode": tri_mode, "tri_max": tri_max})
        elif demand_pattern == "Beta Distribution":
            config.update({"beta_alpha": beta_alpha, "beta_beta": beta_beta, "beta_scale": beta_scale})
        elif demand_pattern == "Log-Normal Distribution":
            config.update({"lognorm_mu": lognorm_mu, "lognorm_sigma": lognorm_sigma})
        elif demand_pattern == "Bimodal Distribution":
            config.update({
                "bimodal_mean1": bimodal_mean1, "bimodal_std1": bimodal_std1,
                "bimodal_mean2": bimodal_mean2, "bimodal_std2": bimodal_std2,
                "bimodal_weight": bimodal_weight
            })
        elif demand_pattern == "Seasonal+Random":
            config.update({
                "seasonal_base": seasonal_base, "seasonal_amplitude": seasonal_amplitude,
                "seasonal_period": seasonal_period, "seasonal_noise": seasonal_noise
            })
        elif demand_pattern == "Trend+Cyclic":
            config.update({
                "trend_base": trend_base, "trend_slope": trend_slope,
                "trend_amplitude": trend_amplitude, "trend_period": trend_period
            })
        elif demand_pattern == "Markov Chain":
            config.update({"markov_states": markov_states, "markov_initial": markov_initial})
        elif demand_pattern == "Promotion-Driven":
            config.update({
                "promo_base": promo_base, "promo_intensity": promo_intensity,
                "promo_start": promo_start, "promo_duration": promo_duration,
                "promo_decay": promo_decay
            })
        elif demand_pattern == "Competition":
            config.update({
                "comp_base": comp_base, "comp_share": comp_share,
                "comp_elasticity": comp_elasticity
            })
        elif demand_pattern == "New Product Diffusion":
            config.update({
                "diffusion_potential": diffusion_potential,
                "diffusion_innovation": diffusion_innovation,
                "diffusion_imitation": diffusion_imitation
            })
        elif demand_pattern == "Inventory Sensitive":
            config.update({
                "inventory_base": inventory_base,
                "inventory_penalty": inventory_penalty,
                "inventory_substitution": inventory_substitution
            })
        
        # Volatile Demand Pattern Parameters
        elif demand_pattern == "Autoregressive":
            config.update({
                "ar_base_demand": ar_base_demand,
                "ar_coeff1": ar_coeff1,
                "ar_coeff2": ar_coeff2,
                "ar_noise_std": ar_noise_std
            })
        elif demand_pattern == "ARMA Demand":
            config.update({
                "arma_base_demand": arma_base_demand,
                "arma_ar_coeff1": arma_ar_coeff1,
                "arma_ar_coeff2": arma_ar_coeff2,
                "arma_ma_coeff1": arma_ma_coeff1,
                "arma_ma_coeff2": arma_ma_coeff2,
                "arma_noise_std": arma_noise_std
            })
        elif demand_pattern == "Jump Diffusion":
            config.update({
                "jump_base_demand": jump_base_demand,
                "jump_drift": jump_drift,
                "jump_volatility": jump_volatility,
                "jump_intensity": jump_intensity,
                "jump_mean": jump_mean,
                "jump_std": jump_std
            })
        elif demand_pattern == "Poisson Jump":
            config.update({
                "poisson_base_demand": poisson_base_demand,
                "poisson_jump_rate": poisson_jump_rate,
                "jump_size_1": jump_size_1,
                "jump_size_2": jump_size_2,
                "jump_size_3": jump_size_3,
                "jump_size_4": jump_size_4
            })
        elif demand_pattern == "Regime Switching":
            config.update({
                "regime_base_demand": regime_base_demand,
                "regime_low_mean": regime_low_mean,
                "regime_low_std": regime_low_std,
                "regime_high_mean": regime_high_mean,
                "regime_high_std": regime_high_std,
                "transition_low_to_high": transition_low_to_high,
                "transition_high_to_low": transition_high_to_low
            })
        elif demand_pattern == "Volatility Clustering":
            config.update({
                "volatility_base_demand": volatility_base_demand,
                "volatility_alpha": volatility_alpha,
                "volatility_beta": volatility_beta,
                "volatility_omega": volatility_omega
            })
        
        # Add Information Sharing Parameters
        if enable_info_sharing:
            config.update({
                "share_inventory": share_inventory,
                "share_demand": share_demand,
                "share_orders": share_orders
            })
        
        # LLM parameters already set in main config dictionary
        
        return config
    
    def _get_pattern_type_from_ui(self, demand_pattern: str) -> str:
        """Convert UI demand pattern name to demand pattern type"""
        pattern_mapping = {
            "Random Demand": "random",
            "Step Demand": "step", 
            "Seasonal Demand": "seasonal",
            "Constant Demand": "constant",
            "Normal Distribution": "normal",
            "Poisson Distribution": "poisson",
            "Exponential Distribution": "exponential",
            "Triangular Distribution": "triangular",
            "Beta Distribution": "beta",
            "Log-Normal Distribution": "lognormal",
            "Bimodal Distribution": "bimodal",
            "Seasonal+Random": "seasonal_random",
            "Trend+Cyclic": "trend_cyclic",
            "Multi-Stage Mix": "multistage",
            "Markov Chain": "markov",
            "Promotion-Driven": "promotion",
            "Competition": "competition",
            "New Product Diffusion": "diffusion",
            "Inventory Sensitive": "inventory_sensitive",
            # Volatile Demand Pattern
            "Autoregressive": "autoregressive",
            "ARMA Demand": "arma",
            "Jump Diffusion": "jump_diffusion",
            "Poisson Jump": "poisson_jump",
            "Regime Switching": "regime_switching",
            "Volatility Clustering": "volatility_clustering"
        }
        return pattern_mapping.get(demand_pattern, "constant")
    
    def _build_distribution_params(self, demand_config: dict) -> dict:
        """Build Distribution Parameter Dictionary"""
        params = {}
        config_type = demand_config.get('type', '')
        
        if config_type == "Normal Distribution":
            params = {
                'mean': demand_config.get('mean', 10.0),
                'std': demand_config.get('std', 2.0)
            }
        elif config_type == "Poisson Distribution":
            params = {
                'lambda': demand_config.get('lambda', 8.0)
            }
        elif config_type == "Exponential Distribution":
            params = {
                'lambda': demand_config.get('lambda', 0.2),
                'base': demand_config.get('base', 3)
            }
        elif config_type == "Triangular Distribution":
            params = {
                'min': demand_config.get('min', 2),
                'mode': demand_config.get('mode', 10),
                'max': demand_config.get('max', 18)
            }
        elif config_type == "Beta Distribution":
            params = {
                'alpha': demand_config.get('alpha', 2.0),
                'beta': demand_config.get('beta', 2.0),
                'scale': demand_config.get('scale', 15)
            }
        elif config_type == "Log-Normal Distribution":
            params = {
                'mu': demand_config.get('mu', 2.0),
                'sigma': demand_config.get('sigma', 0.5)
            }
        elif config_type == "Bimodal Distribution":
            params = {
                'mean1': demand_config.get('mean1', 5.0),
                'std1': demand_config.get('std1', 1.0),
                'mean2': demand_config.get('mean2', 15.0),
                'std2': demand_config.get('std2', 2.0),
                'weight': demand_config.get('weight', 0.6)
            }
        elif config_type == "Seasonal+Random":
            params = {
                'base': demand_config.get('base', 10.0),
                'amplitude': demand_config.get('amplitude', 3.0),
                'period': demand_config.get('period', 12),
                'noise_std': demand_config.get('noise_std', 1.5)
            }
        elif config_type == "Trend+Cyclic":
            params = {
                'base': demand_config.get('base', 8.0),
                'slope': demand_config.get('slope', 0.1),
                'amplitude': demand_config.get('amplitude', 2.0),
                'period': demand_config.get('period', 8)
            }
        elif config_type == "Promotion-Driven":
            params = {
                'base_demand': demand_config.get('base', 8.0),
                'intensity': demand_config.get('intensity', 3.0),
                'start_week': demand_config.get('start_week', 10),
                'duration': demand_config.get('duration', 3),
                'decay_rate': demand_config.get('decay_rate', 0.5)
            }
        elif config_type == "Competition":
            params = {
                'base_demand': demand_config.get('base', 10.0),
                'market_share': demand_config.get('market_share', 0.4),
                'elasticity': demand_config.get('elasticity', 1.5),
                'competitor_actions': []
            }
        elif config_type == "New Product Diffusion":
            params = {
                'market_potential': demand_config.get('market_potential', 1000),
                'innovation_coeff': demand_config.get('innovation_coeff', 0.03),
                'imitation_coeff': demand_config.get('imitation_coeff', 0.38)
            }
        elif config_type == "Inventory Sensitive":
            params = {
                'base_demand': demand_config.get('base', 10.0),
                'stockout_penalty': demand_config.get('stockout_penalty', 0.3),
                'substitution_rate': demand_config.get('substitution_rate', 0.2)
            }
        
        # Volatile Demand Pattern Parameters
        elif config_type == "Autoregressive":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'ar_coefficients': demand_config.get('ar_coefficients', [0.7, -0.2]),
                'noise_std': demand_config.get('noise_std', 5.0)
            }
        elif config_type == "ARMA Demand":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'ar_coefficients': demand_config.get('ar_coefficients', [0.5, 0.3]),
                'ma_coefficients': demand_config.get('ma_coefficients', [0.4, -0.2]),
                'noise_std': demand_config.get('noise_std', 8.0)
            }
        elif config_type == "Jump Diffusion":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'drift': demand_config.get('drift', 0.02),
                'volatility': demand_config.get('volatility', 0.15),
                'jump_intensity': demand_config.get('jump_intensity', 0.1),
                'jump_mean': demand_config.get('jump_mean', 0.0),
                'jump_std': demand_config.get('jump_std', 0.3)
            }
        elif config_type == "Poisson Jump":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'jump_rate': demand_config.get('jump_rate', 0.3),
                'jump_sizes': demand_config.get('jump_sizes', [-10, -5, 5, 15])
            }
        elif config_type == "Regime Switching":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'regime_means': [demand_config.get('low_mean', 80), demand_config.get('high_mean', 120)],
                'regime_stds': [demand_config.get('low_std', 5), demand_config.get('high_std', 8)],
                'transition_matrix': [
                    [1 - demand_config.get('low_to_high', 0.15), demand_config.get('low_to_high', 0.15)],
                    [demand_config.get('high_to_low', 0.20), 1 - demand_config.get('high_to_low', 0.20)]
                ]
            }
        elif config_type == "Volatility Clustering":
            params = {
                'base_demand': demand_config.get('base_demand', 100),
                'alpha': demand_config.get('alpha', 0.15),
                'beta': demand_config.get('beta', 0.75),
                'omega': demand_config.get('omega', 2.0)
            }
        
        return params
    
    def create_game_config(self, ui_config: Dict[str, Any]):
        """Based on UI Config, Create Game Config"""
        config = get_default_config()
        
        # Update Config Parameters
        config.simulation.total_weeks = ui_config["num_rounds"]
        config.simulation.random_seed = ui_config["random_seed"]
        
        # Update Cost Parameters
        cost_configs = ui_config.get("cost_configs", {})
        individual_costs = cost_configs.get("individual", {})
        
        for role in ["retailer", "wholesaler", "distributor", "manufacturer"]:
            role_config = getattr(config, role)
            
            # Apply independent cost parameters
            if role in individual_costs:
                role_costs = individual_costs[role]
                role_config.cost_config.holding_cost = role_costs.get("holding_cost", 0.5)
                role_config.cost_config.backorder_cost = role_costs.get("shortage_cost", 1.0)
                role_config.cost_config.order_cost = 0.0
                print(f"💰 {role} Cost Config: Holding={role_config.cost_config.holding_cost}, Stockout={role_config.cost_config.backorder_cost}")
            else:
                # Use default values as fallback
                role_config.cost_config.holding_cost = 0.5
                role_config.cost_config.backorder_cost = 1.0
                role_config.cost_config.order_cost = 0.0
                print(f"⚠️ {role} Using default cost config")
            
            # Map initial inventory and initial in-transit values
            # Initial Inventory (whole number)
            if f"{role}_initial_inventory" in ui_config:
                try:
                    role_config.initial_inventory = int(ui_config[f"{role}_initial_inventory"]) if ui_config[f"{role}_initial_inventory"] is not None else role_config.initial_inventory
                except Exception:
                    pass  # Maintain default
            # Initial In-Transit（List）
            if f"{role}_initial_in_transit" in ui_config:
                init_transit = ui_config.get(f"{role}_initial_in_transit")
                # Only parse when list is set, already parsed in UI stage
                if isinstance(init_transit, list):
                    # Filter negative numbers, ensure non-negative whole numbers
                    role_config.initial_in_transit = [max(0, int(x)) for x in init_transit if isinstance(x, (int, float))]
                else:
                    # If empty string or None, keep default
                    pass
            
            # Set Order Quantity Limit (ensure always set, avoid config errors)
            # Handle Discrete Point Select
            if ui_config.get("enable_discrete_points", False):
                # Set Discrete Point Select parameters
                role_config.enable_discrete_points = True
                discrete_points_str = ui_config.get(f"{role}_discrete_points", "4,8,12,16,20")
                if isinstance(discrete_points_str, str):
                    # Parse and filter out 0 and negative numbers
                    role_config.discrete_order_points = [int(x.strip()) for x in discrete_points_str.split(',') if x.strip() and int(x.strip()) > 0]
                else:
                    # If list, also need to filter out 0 and negative numbers
                    role_config.discrete_order_points = [p for p in discrete_points_str if p > 0]
                print(f"✅ {role} Discrete Order Points enabled: {role_config.discrete_order_points}")
            else:
                role_config.enable_discrete_points = False
                role_config.discrete_order_points = None
            
            # Handle Min / Max Order Quantity Limit
            if ui_config.get("enable_order_limits", False):
                # Set continuous range limit parameters
                role_config.min_order_quantity = ui_config.get(f"{role}_min_order", 3)
                role_config.max_order_quantity = ui_config.get(f"{role}_max_order", 8)
                print(f"✅ {role} Order Limit enabled: [{role_config.min_order_quantity}, {role_config.max_order_quantity}]")
            else:
                # Disabled order limit: set to extreme range
                role_config.min_order_quantity = 0
                role_config.max_order_quantity = 1000
                print(f"❌ {role} Order Limit disabled: no limit")
                
        # Add adjust/debug info: display final config status
        print(f"\n🔧 Final Config Status:")
        print(f"   Order Limit Enable: {ui_config.get('enable_order_limits', False)}")
        for role in ["retailer", "wholesaler", "distributor", "manufacturer"]:
            role_config = getattr(config, role)
            print(f"   {role}: [{role_config.min_order_quantity}, {role_config.max_order_quantity}]")
        
        # Update Demand Pattern
        demand_config = config.demand
        pattern = ui_config["demand_pattern"]
        
        # Basic Distribution Types
        if pattern == "Random Demand":
            demand_config.pattern_type = DemandPatternType.RANDOM
            demand_config.random_min = ui_config["random_min"]
            demand_config.random_max = ui_config["random_max"]
            print(f"📊 Random Demand: [{demand_config.random_min}, {demand_config.random_max}]")
        elif pattern == "Step Demand":
            demand_config.pattern_type = DemandPatternType.STEP
            demand_config.base_demand = ui_config["initial_demand"]
            demand_config.step_demand = ui_config["step_demand"]
            demand_config.step_week = ui_config["step_week"]
        elif pattern == "Seasonal Demand":
            demand_config.pattern_type = DemandPatternType.SEASONAL
            demand_config.base_demand = ui_config["base_demand"]
            demand_config.seasonal_amplitude = ui_config["amplitude"]
            demand_config.seasonal_period = ui_config["period"]
        elif pattern == "Constant Demand":
            demand_config.pattern_type = DemandPatternType.STEP  # Use STEP as Constant Demand
            demand_config.base_demand = ui_config.get("initial_demand", 4)
        
        # Common Distribution Types
        elif pattern == "Normal Distribution":
            demand_config.pattern_type = DemandPatternType.NORMAL
            demand_config.normal_mean = ui_config["normal_mean"]
            demand_config.normal_std = ui_config["normal_std"]
        elif pattern == "Poisson Distribution":
            demand_config.pattern_type = DemandPatternType.POISSON
            demand_config.poisson_lambda = ui_config["poisson_lambda"]
        elif pattern == "Exponential Distribution":
            demand_config.pattern_type = DemandPatternType.EXPONENTIAL
            demand_config.exponential_lambda = ui_config["exp_lambda"]
            demand_config.exponential_base = ui_config["exp_base"]
        elif pattern == "Triangular Distribution":
            demand_config.pattern_type = DemandPatternType.TRIANGULAR
            demand_config.triangular_min = ui_config["tri_min"]
            demand_config.triangular_mode = ui_config["tri_mode"]
            demand_config.triangular_max = ui_config["tri_max"]
        elif pattern == "Beta Distribution":
            demand_config.pattern_type = DemandPatternType.BETA
            demand_config.beta_alpha = ui_config["beta_alpha"]
            demand_config.beta_beta = ui_config["beta_beta"]
            demand_config.beta_scale = ui_config["beta_scale"]
        elif pattern == "Log-Normal Distribution":
            demand_config.pattern_type = DemandPatternType.LOGNORMAL
            demand_config.lognormal_mu = ui_config["lognorm_mu"]
            demand_config.lognormal_sigma = ui_config["lognorm_sigma"]
        
        # Mixed Distribution Types
        elif pattern == "Bimodal Distribution":
            demand_config.pattern_type = DemandPatternType.BIMODAL
            demand_config.bimodal_mean1 = ui_config["bimodal_mean1"]
            demand_config.bimodal_std1 = ui_config["bimodal_std1"]
            demand_config.bimodal_mean2 = ui_config["bimodal_mean2"]
            demand_config.bimodal_std2 = ui_config["bimodal_std2"]
            demand_config.bimodal_weight = ui_config["bimodal_weight"]
        elif pattern == "Seasonal+Random":
            demand_config.pattern_type = DemandPatternType.SEASONAL_RANDOM
            demand_config.seasonal_random_base = ui_config["seasonal_base"]
            demand_config.seasonal_random_amplitude = ui_config["seasonal_amplitude"]
            demand_config.seasonal_random_period = ui_config["seasonal_period"]
            demand_config.seasonal_random_noise_std = ui_config["seasonal_noise"]
        elif pattern == "Trend+Cyclic":
            demand_config.pattern_type = DemandPatternType.TREND_CYCLIC
            demand_config.trend_cyclic_base = ui_config["trend_base"]
            demand_config.trend_cyclic_slope = ui_config["trend_slope"]
            demand_config.trend_cyclic_amplitude = ui_config["trend_amplitude"]
            demand_config.trend_cyclic_period = ui_config["trend_period"]
        elif pattern == "Multi-Stage Mix":
            demand_config.pattern_type = DemandPatternType.MULTISTAGE
            # Use preset multi-stage config
            demand_config.multistage_stages = [
                {"duration": 10, "type": "normal", "mean": 8, "std": 1.5},
                {"duration": 10, "type": "exponential", "lambda": 0.15, "base": 5},
                {"duration": 10, "type": "constant", "value": 12}
            ]
        elif pattern == "Markov Chain":
            demand_config.pattern_type = DemandPatternType.MARKOV
            # Parse user input status
            states_str = ui_config.get("markov_states", "3,8,15")
            demand_config.markov_states = [int(x.strip()) for x in states_str.split(",")]
            demand_config.markov_initial_state = ui_config.get("markov_initial", 1)
            # Use preset transition matrix
            demand_config.markov_transition_matrix = [
                [0.7, 0.2, 0.1],
                [0.3, 0.5, 0.2],
                [0.1, 0.3, 0.6]
            ]
        
        # Real Scenario Types
        elif pattern == "Promotion-Driven":
            demand_config.pattern_type = DemandPatternType.PROMOTION
            demand_config.promotion_base_demand = ui_config["promo_base"]
            demand_config.promotion_intensity = ui_config["promo_intensity"]
            demand_config.promotion_start_week = ui_config["promo_start"]
            demand_config.promotion_duration = ui_config["promo_duration"]
            demand_config.promotion_decay_rate = ui_config["promo_decay"]
        elif pattern == "Competition":
            demand_config.pattern_type = DemandPatternType.COMPETITION
            demand_config.competition_base_demand = ui_config["comp_base"]
            demand_config.competition_market_share = ui_config["comp_share"]
            demand_config.competition_elasticity = ui_config["comp_elasticity"]
            # Use preset competition events
            demand_config.competition_competitor_actions = [
                {"week": 8, "action": "price_cut", "intensity": 0.2},
                {"week": 15, "action": "promotion", "intensity": 0.3}
            ]
        elif pattern == "New Product Diffusion":
            demand_config.pattern_type = DemandPatternType.DIFFUSION
            demand_config.diffusion_market_potential = ui_config["diffusion_potential"]
            demand_config.diffusion_innovation_coeff = ui_config["diffusion_innovation"]
            demand_config.diffusion_imitation_coeff = ui_config["diffusion_imitation"]
        elif pattern == "Inventory Sensitive":
            demand_config.pattern_type = DemandPatternType.INVENTORY_SENSITIVE
            demand_config.inventory_sensitive_base_demand = ui_config["inventory_base"]
            demand_config.inventory_sensitive_stockout_penalty = ui_config["inventory_penalty"]
            demand_config.inventory_sensitive_substitution_rate = ui_config["inventory_substitution"]
        
        # VolatileDemand Pattern
        elif pattern == "Autoregressive":
            demand_config.pattern_type = DemandPatternType.AUTOREGRESSIVE
            demand_config.ar_base_demand = ui_config["ar_base_demand"]
            demand_config.ar_coefficients = [ui_config["ar_coeff1"], ui_config["ar_coeff2"]]
            demand_config.ar_noise_std = ui_config["ar_noise_std"]
        elif pattern == "ARMA Demand":
            demand_config.pattern_type = DemandPatternType.ARMA
            demand_config.arma_base_demand = ui_config["arma_base_demand"]
            demand_config.arma_ar_coefficients = [ui_config["arma_ar_coeff1"], ui_config["arma_ar_coeff2"]]
            demand_config.arma_ma_coefficients = [ui_config["arma_ma_coeff1"], ui_config["arma_ma_coeff2"]]
            demand_config.arma_noise_std = ui_config["arma_noise_std"]
        elif pattern == "Jump Diffusion":
            demand_config.pattern_type = DemandPatternType.JUMP_DIFFUSION
            demand_config.jump_base_demand = ui_config["jump_base_demand"]
            demand_config.jump_drift = ui_config["jump_drift"]
            demand_config.jump_volatility = ui_config["jump_volatility"]
            demand_config.jump_intensity = ui_config["jump_intensity"]
            demand_config.jump_mean = ui_config["jump_mean"]
            demand_config.jump_std = ui_config["jump_std"]
        elif pattern == "Poisson Jump":
            demand_config.pattern_type = DemandPatternType.POISSON_JUMP
            demand_config.poisson_base_demand = ui_config["poisson_base_demand"]
            demand_config.poisson_jump_rate = ui_config["poisson_jump_rate"]
            demand_config.poisson_jump_sizes = [
                ui_config["jump_size_1"], ui_config["jump_size_2"],
                ui_config["jump_size_3"], ui_config["jump_size_4"]
            ]
        elif pattern == "Regime Switching":
            demand_config.pattern_type = DemandPatternType.REGIME_SWITCHING
            demand_config.regime_base_demand = ui_config["regime_base_demand"]
            demand_config.regime_low_mean = ui_config["regime_low_mean"]
            demand_config.regime_low_std = ui_config["regime_low_std"]
            demand_config.regime_high_mean = ui_config["regime_high_mean"]
            demand_config.regime_high_std = ui_config["regime_high_std"]
            demand_config.regime_transition_low_to_high = ui_config["transition_low_to_high"]
            demand_config.regime_transition_high_to_low = ui_config["transition_high_to_low"]
        elif pattern == "Volatility Clustering":
            demand_config.pattern_type = DemandPatternType.VOLATILITY_CLUSTERING
            demand_config.volatility_base_demand = ui_config["volatility_base_demand"]
            demand_config.volatility_alpha = ui_config["volatility_alpha"]
            demand_config.volatility_beta = ui_config["volatility_beta"]
            demand_config.volatility_omega = ui_config["volatility_omega"]
        
        print(f"📊 Demand Pattern already set: {pattern}")
        
        # Update Information Sharing
        if ui_config["enable_info_sharing"]:
            config.simulation.information_sharing = True
            if "share_inventory" in ui_config:
                config.simulation.shared_inventory_levels = ui_config["share_inventory"]
            if "share_demand" in ui_config:
                config.simulation.shared_demand_history = ui_config["share_demand"]
        
        # Update LLM Config
        llm_provider = ui_config.get("llm_provider", "Mock LLM")
        # Provider label -> internal provider key
        _provider_key_map = {
            "Mock LLM": "mock", "OpenAI": "openai", "Anthropic": "anthropic",
            "Ollama": "ollama", "GPT-OSS": "gpt_oss",
            "DeepSeek": "deepseek", "Zhipu GLM": "zhipu",
            "Moonshot Kimi": "moonshot", "Qwen": "qwen",
            "OpenRouter": "openrouter", "Groq": "groq",
            "Together AI": "together", "Custom OpenAI-Compatible": "openai_compatible",
        }
        _needs_api_key = {"OpenAI", "Anthropic", "DeepSeek", "Zhipu GLM", "Moonshot Kimi",
                          "Qwen", "OpenRouter", "Groq", "Together AI"}
        config.llm.provider = _provider_key_map.get(llm_provider, "mock")
        if ui_config.get("model_name"):
            config.llm.model = ui_config["model_name"]
        if llm_provider in _needs_api_key and ui_config.get("api_key"):
            config.llm.api_key = ui_config["api_key"]
        if ui_config.get("base_url"):
            config.llm.base_url = ui_config["base_url"]

        # Apply preset max_tokens, timeout, and thinking model flag
        if ui_config.get("llm_max_tokens"):
            config.llm.max_tokens = ui_config["llm_max_tokens"]
        if ui_config.get("llm_timeout"):
            config.llm.timeout = ui_config["llm_timeout"]
        if ui_config.get("llm_is_thinking_model"):
            config.llm.is_thinking_model = ui_config["llm_is_thinking_model"]
        if "enable_demand_forecasting" in ui_config:
            config.enable_demand_forecasting = ui_config["enable_demand_forecasting"]

        # Update Lead Time Config
        if "retailer_order_lead_time" in ui_config:
            config.simulation.retailer_lead_time = ui_config["retailer_order_lead_time"]
        if "wholesaler_order_lead_time" in ui_config:
            config.simulation.wholesaler_lead_time = ui_config["wholesaler_order_lead_time"]
        if "distributor_order_lead_time" in ui_config:
            config.simulation.distributor_lead_time = ui_config["distributor_order_lead_time"]
        # Manufacturer Lead Time Mapping: Priority use “Production Lead Time”; only override when “Order Lead Time” > 0 (avoid default 0 false override)
        if "manufacturer_production_lead_time" in ui_config:
            config.simulation.manufacturer_lead_time = ui_config["manufacturer_production_lead_time"]
        if "manufacturer_order_lead_time" in ui_config and ui_config["manufacturer_order_lead_time"] and ui_config["manufacturer_order_lead_time"] > 0:
            config.simulation.manufacturer_lead_time = ui_config["manufacturer_order_lead_time"]
        # Lower limit protection, avoid showing 0 in prompt display (use max(1,0))
        try:
            config.simulation.manufacturer_lead_time = max(1, int(config.simulation.manufacturer_lead_time))
        except Exception:
            config.simulation.manufacturer_lead_time = 2
        
        # Note: Transport lead time parameter temporarily stored in config for future extension use
        config.retailer_transport_lead_time = ui_config.get("retailer_transport_lead_time", 2)
        config.wholesaler_transport_lead_time = ui_config.get("wholesaler_transport_lead_time", 2)
        config.distributor_transport_lead_time = ui_config.get("distributor_transport_lead_time", 2)
        
        # AddCoordinator Config
        # AddAdaptiveOrder Quantity Limit Config
        if ui_config.get("enable_adaptive_limits", False):
            # EnableAdaptiveLimit
            config.adaptive_limits.enabled = True
            config.adaptive_limits.method = ui_config.get("adaptive_method", "ratio")
            config.adaptive_limits.base_min_order = ui_config.get("adaptive_base_min", 3)
            config.adaptive_limits.base_max_order = ui_config.get("adaptive_base_max", 8)
            
            # Based onSelectMethodSetParameter
            if config.adaptive_limits.method == "ratio":
                config.adaptive_limits.min_ratio = ui_config.get("min_ratio", 0.5)
                config.adaptive_limits.max_ratio = ui_config.get("max_ratio", 1.5)
            elif config.adaptive_limits.method == "window":
                config.adaptive_limits.window_size = ui_config.get("window_size", 5)
                config.adaptive_limits.alpha = ui_config.get("alpha", 1.0)
                config.adaptive_limits.beta = ui_config.get("beta", 2.0)
            elif config.adaptive_limits.method == "forecast":
                config.adaptive_limits.forecast_horizon = ui_config.get("forecast_horizon", 3)
                config.adaptive_limits.forecast_weight = ui_config.get("forecast_weight", 0.7)
            
            # SetGeneralParameter
            config.adaptive_limits.adaptation_rate = ui_config.get("adaptation_rate", 0.3)
            
            print(f"✅ Adaptive Order Quantity Limit enabled: Method={config.adaptive_limits.method}")
            print(f"   Base Limit Range: [{config.adaptive_limits.base_min_order}, {config.adaptive_limits.base_max_order}]")
        else:
            # DisabledAdaptiveLimit
            config.adaptive_limits.enabled = False
            print(f"❌ Adaptive Order Quantity Limit disabled")
        
        return config
    
    def _build_demand_config(self, demand_pattern: str, demand_category: str, local_vars: dict) -> dict:
        """Build Demand Config Dictionary"""
        config = {
            'type': demand_pattern,
            'category': demand_category
        }
        
        # Based onDemand TypeAddCorrespondingParameter
        if demand_pattern == "Random Demand":
            config.update({
                'min_demand': local_vars.get('random_min', 1),
                'max_demand': local_vars.get('random_max', 10)
            })
        elif demand_pattern == "Step Demand":
            config.update({
                'initial_demand': local_vars.get('initial_demand', 4),
                'step_demand': local_vars.get('step_demand', 8),
                'step_week': local_vars.get('step_week', 5)
            })
        elif demand_pattern == "Seasonal Demand":
            config.update({
                'base_demand': local_vars.get('base_demand', 25),
                'amplitude': local_vars.get('amplitude', 10),
                'period': local_vars.get('period', 12)
            })
        elif demand_pattern == "Constant Demand":
            config.update({
                'demand': local_vars.get('initial_demand', 4)
            })
        elif demand_pattern == "Normal Distribution":
            config.update({
                'mean': local_vars.get('normal_mean', 10.0),
                'std': local_vars.get('normal_std', 2.0)
            })
        elif demand_pattern == "Poisson Distribution":
            config.update({
                'lambda': local_vars.get('poisson_lambda', 8.0)
            })
        elif demand_pattern == "Exponential Distribution":
            config.update({
                'lambda': local_vars.get('exp_lambda', 0.2),
                'base': local_vars.get('exp_base', 3)
            })
        elif demand_pattern == "Triangular Distribution":
            config.update({
                'min': local_vars.get('tri_min', 2),
                'mode': local_vars.get('tri_mode', 10),
                'max': local_vars.get('tri_max', 18)
            })
        elif demand_pattern == "Beta Distribution":
            config.update({
                'alpha': local_vars.get('beta_alpha', 2.0),
                'beta': local_vars.get('beta_beta', 2.0),
                'scale': local_vars.get('beta_scale', 15)
            })
        elif demand_pattern == "Log-Normal Distribution":
            config.update({
                'mu': local_vars.get('lognorm_mu', 2.0),
                'sigma': local_vars.get('lognorm_sigma', 0.5)
            })
        elif demand_pattern == "Bimodal Distribution":
            config.update({
                'mean1': local_vars.get('bimodal_mean1', 5.0),
                'std1': local_vars.get('bimodal_std1', 1.0),
                'mean2': local_vars.get('bimodal_mean2', 15.0),
                'std2': local_vars.get('bimodal_std2', 2.0),
                'weight': local_vars.get('bimodal_weight', 0.6)
            })
        elif demand_pattern == "Seasonal+Random":
            config.update({
                'base': local_vars.get('seasonal_base', 10.0),
                'amplitude': local_vars.get('seasonal_amplitude', 3.0),
                'period': local_vars.get('seasonal_period', 12),
                'noise_std': local_vars.get('seasonal_noise', 1.5)
            })
        elif demand_pattern == "Trend+Cyclic":
            config.update({
                'base': local_vars.get('trend_base', 8.0),
                'slope': local_vars.get('trend_slope', 0.1),
                'amplitude': local_vars.get('trend_amplitude', 2.0),
                'period': local_vars.get('trend_period', 8)
            })
        elif demand_pattern == "Promotion-Driven":
            config.update({
                'base': local_vars.get('promo_base', 8.0),
                'intensity': local_vars.get('promo_intensity', 3.0),
                'start_week': local_vars.get('promo_start', 10),
                'duration': local_vars.get('promo_duration', 3),
                'decay_rate': local_vars.get('promo_decay', 0.5)
            })
        elif demand_pattern == "Competition":
            config.update({
                'base': local_vars.get('comp_base', 10.0),
                'market_share': local_vars.get('comp_share', 0.4),
                'elasticity': local_vars.get('comp_elasticity', 1.5)
            })
        elif demand_pattern == "New Product Diffusion":
            config.update({
                'market_potential': local_vars.get('diffusion_potential', 1000),
                'innovation_coeff': local_vars.get('diffusion_innovation', 0.03),
                'imitation_coeff': local_vars.get('diffusion_imitation', 0.38)
            })
        elif demand_pattern == "Inventory Sensitive":
            config.update({
                'base': local_vars.get('inventory_base', 10.0),
                'stockout_penalty': local_vars.get('inventory_penalty', 0.3),
                'substitution_rate': local_vars.get('inventory_substitution', 0.2)
            })
        
        return config
    
    def run_simulation(self, config, realtime_callback=None, progress_container=None, show_realtime_status=False) -> tuple:
        """Run Simulation"""
        try:
            # SetRandom Seed
            set_random_seed(config.simulation.random_seed)
            
            # DisplaySimulationStartInfo
            if progress_container:
                # AddCustomCSSStyle
                progress_container.markdown("""
                <style>
                .progress-header {
                    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 24px;
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .simulation-info {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .progress-container {
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 15px;
                    border: 2px solid #e9ecef;
                    margin-bottom: 15px;
                }
                .progress-stats {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    font-weight: bold;
                }
                </style>
                """, unsafe_allow_html=True)
                
                progress_container.markdown('<div class="progress-header">🚀 Simulation in Progress</div>', unsafe_allow_html=True)
                progress_container.markdown(f'''
                <div class="simulation-info">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 18px;">📋 Simulation Config</span><br>
                            <span style="font-size: 14px; opacity: 0.9;">Total Rounds: {config.simulation.total_weeks} | Demand Pattern: {config.demand.pattern_type.value}</span>
                        </div>
                        <div style="font-size: 32px;">🎮</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # CreateProgressDisplayArea
                progress_container.markdown('<div class="progress-container">', unsafe_allow_html=True)
                
                # AddOverallProgressItem
                total_weeks = config.simulation.total_weeks
                progress_stats_container = progress_container.empty()
                overall_progress = progress_container.progress(0)
                overall_status = progress_container.empty()
                
                # InitializeProgressStatistics
                progress_stats_container.markdown(f'''
                <div class="progress-stats">
                    <span>🎯 Progress: 0/{total_weeks} Round</span>
                    <span>⏱️ Status: Ready to Start</span>
                    <span>📊 Completion Rate: 0.0%</span>
                </div>
                ''', unsafe_allow_html=True)
                
                overall_status.info("🔄 Initializing Simulation Environment......")
            
            # Create LLM Manager
            # OpenAI-compatible providers (use OpenAIClient with custom base_url)
            _openai_compat_providers = {"openai", "deepseek", "zhipu", "moonshot", "qwen",
                                        "openrouter", "groq", "together", "openai_compatible"}
            if config.llm.provider == "mock":
                llm_manager = LLMManager([MockLLMClient()])
            elif config.llm.provider in _openai_compat_providers:
                if config.llm.api_key:
                    client = OpenAIClient(
                        api_key=config.llm.api_key,
                        model_name=config.llm.model,
                        base_url=config.llm.base_url
                    )
                    llm_manager = LLMManager([client])
                else:
                    st.error(f"Please provide API Key for provider: {config.llm.provider}")
                    return None, None
            elif config.llm.provider == "anthropic":
                if config.llm.api_key:
                    client = AnthropicClient(api_key=config.llm.api_key, model_name=config.llm.model)
                    llm_manager = LLMManager([client])
                else:
                    st.error("Please provide Anthropic API Key")
                    return None, None
            elif config.llm.provider == "ollama":
                client = OllamaClient(model_name=config.llm.model, base_url=config.llm.base_url,
                                     timeout=config.llm.timeout)
                llm_manager = LLMManager([client])
            elif config.llm.provider == "gpt_oss":
                client = GPTOSSClient(model_name=config.llm.model, base_url=config.llm.base_url,
                                     timeout=config.llm.timeout)
                llm_manager = LLMManager([client])
            else:
                llm_manager = LLMManager([MockLLMClient()])
            
            # CreateSupply ChainAgent
            agents = create_supply_chain(config, llm_client=llm_manager)
            
            # Based onConfigCreate Demand Pattern
            dc = config.demand
            ptype = dc.pattern_type.value

            # Build distribution_params for distribution-based demand types
            dist_params = {}
            if ptype == 'normal':
                dist_params = {'mean': dc.normal_mean, 'std': dc.normal_std}
            elif ptype == 'poisson':
                dist_params = {'lambda': dc.poisson_lambda}
            elif ptype == 'exponential':
                dist_params = {'lambda': dc.exponential_lambda, 'base': dc.exponential_base}
            elif ptype == 'triangular':
                dist_params = {'min': dc.triangular_min, 'max': dc.triangular_max, 'mode': dc.triangular_mode}
            elif ptype == 'beta':
                dist_params = {'alpha': dc.beta_alpha, 'beta': dc.beta_beta, 'scale': dc.beta_scale, 'shift': dc.beta_shift}
            elif ptype == 'lognormal':
                dist_params = {'mu': dc.lognormal_mu, 'sigma': dc.lognormal_sigma}
            elif ptype == 'bimodal':
                dist_params = {'mean1': dc.bimodal_mean1, 'std1': dc.bimodal_std1, 'mean2': dc.bimodal_mean2, 'std2': dc.bimodal_std2, 'weight': dc.bimodal_weight}
            elif ptype == 'seasonal_random':
                dist_params = {'base': dc.seasonal_random_base, 'amplitude': dc.seasonal_random_amplitude, 'period': dc.seasonal_random_period, 'noise_std': dc.seasonal_random_noise_std}
            elif ptype == 'trend_cyclic':
                dist_params = {'base': dc.trend_cyclic_base, 'slope': dc.trend_cyclic_slope, 'amplitude': dc.trend_cyclic_amplitude, 'period': dc.trend_cyclic_period}
            elif ptype == 'multistage':
                dist_params = {'stages': dc.multistage_stages}
            elif ptype == 'markov':
                dist_params = {'states': dc.markov_states, 'transition_matrix': dc.markov_transition_matrix, 'initial_state': dc.markov_initial_state}
            elif ptype == 'promotion':
                dist_params = {'base_demand': dc.promotion_base_demand, 'intensity': dc.promotion_intensity, 'start_week': dc.promotion_start_week, 'duration': dc.promotion_duration, 'decay_rate': dc.promotion_decay_rate}
            elif ptype == 'competition':
                dist_params = {'base_demand': dc.competition_base_demand, 'market_share': dc.competition_market_share, 'elasticity': dc.competition_elasticity, 'competitor_actions': dc.competition_competitor_actions}
            elif ptype == 'diffusion':
                dist_params = {'market_potential': dc.diffusion_market_potential, 'innovation_coeff': dc.diffusion_innovation_coeff, 'imitation_coeff': dc.diffusion_imitation_coeff}
            elif ptype == 'inventory_sensitive':
                dist_params = {'base_demand': dc.inventory_sensitive_base_demand, 'stockout_penalty': dc.inventory_sensitive_stockout_penalty, 'substitution_rate': dc.inventory_sensitive_substitution_rate}
            elif ptype == 'autoregressive':
                dist_params = {'base_demand': getattr(dc, 'ar_base_demand', dc.base_demand), 'ar_coeffs': getattr(dc, 'ar_coefficients', [0.7]), 'noise_std': getattr(dc, 'ar_noise_std', 2.0)}
            elif ptype == 'arma':
                dist_params = {'base_demand': getattr(dc, 'arma_base_demand', dc.base_demand), 'ar_coeffs': getattr(dc, 'arma_ar_coefficients', [0.5]), 'ma_coeffs': getattr(dc, 'arma_ma_coefficients', [0.3]), 'noise_std': getattr(dc, 'arma_noise_std', 2.0)}
            elif ptype == 'jump_diffusion':
                dist_params = {'base_demand': getattr(dc, 'jump_base_demand', dc.base_demand), 'drift': getattr(dc, 'jump_drift', 0.0), 'volatility': getattr(dc, 'jump_volatility', 0.2), 'jump_intensity': getattr(dc, 'jump_intensity', 0.1), 'jump_mean': getattr(dc, 'jump_mean', 0.0), 'jump_std': getattr(dc, 'jump_std', 0.5)}
            elif ptype == 'poisson_jump':
                dist_params = {'base_demand': getattr(dc, 'poisson_base_demand', dc.base_demand), 'jump_rate': getattr(dc, 'poisson_jump_rate', 0.2), 'jump_sizes': getattr(dc, 'poisson_jump_sizes', [-5, -2, 3, 8]), 'jump_probs': getattr(dc, 'poisson_jump_probs', [0.2, 0.3, 0.3, 0.2])}
            elif ptype == 'regime_switching':
                dist_params = {'regimes': getattr(dc, 'switching_regimes', [{'mean': 8, 'std': 2, 'name': 'low'}, {'mean': 15, 'std': 3, 'name': 'high'}]), 'transition_matrix': getattr(dc, 'switching_transition_matrix', [[0.9, 0.1], [0.15, 0.85]])}
            elif ptype == 'volatility_clustering':
                dist_params = {'base_demand': getattr(dc, 'volatility_base_demand', dc.base_demand), 'alpha': getattr(dc, 'volatility_alpha', 0.1), 'beta': getattr(dc, 'volatility_beta', 0.8), 'omega': getattr(dc, 'volatility_omega', 1.0)}

            demand_pattern = DemandPattern(
                pattern_type=ptype,
                base_demand=dc.base_demand,
                step_change=getattr(dc, 'step_demand', dc.base_demand) - dc.base_demand,
                step_round=getattr(dc, 'step_week', 5),
                seasonal_amplitude=getattr(dc, 'seasonal_amplitude', 0.0),
                seasonal_period=getattr(dc, 'seasonal_period', 12),
                random_min=getattr(dc, 'random_min', 1),
                random_max=getattr(dc, 'random_max', 10),
                distribution_params=dist_params,
            )
            

            
            # CreateGameEngine
            engine = GameEngine(
                config,
                agents,
                demand_pattern=demand_pattern,
                enable_coordinator=False,
                llm_client=None
            )
            
            # EnsureGameStatusfullyReSet
            engine.reset()
            
            # CreateProgressCallback FuncNumber
            def progress_callback(round_num, round_data):
                nonlocal progress_stats_container, overall_progress, overall_status
                print(f"🔄 Progress callback called: Round {round_num}")  # AdjustTryOutput
                sys.stdout.flush()  # StrongProductionRefreshNewOutput
                if progress_container:
                    with progress_container:
                        # UpdateOverallProgressItem
                        progress_percentage = round_num / total_weeks
                        overall_progress.progress(progress_percentage)
                        
                        # Usenative componentsDisplayProgressStatistics
                        progress_stats_container.info(f"🎯 Progress: {round_num}/{total_weeks} Round | ⏱️ Status: Runin | 📊 Completion Rate: {progress_percentage*100:.1f}%")
                        
                        # DisplayCurrentRoundInfo
                        overall_status.success(f"🎮 Running Round {round_num} - Customer Demand: {round_data['customer_demand']} | Round Cost: {round_data['total_cost']:.1f}")
                        
                        # Only display each role's real-time status when user selects the option
                        if show_realtime_status:
                            # Use collapsible expander to display each role's real-time status
                            with st.expander("📊 Each Role Real-time Status", expanded=False):
                                # CreateAgentStatusDisplay
                                role_names = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}
                                role_icons = {'retailer': '🏪', 'wholesaler': '🏢', 'distributor': '🚚', 'manufacturer': '🏭'}
                                
                                # UseColumnLayoutDisplayAgentStatus
                                cols = st.columns(2)
                                col_idx = 0
                                
                                for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                                    if role in round_data['agents']:
                                        agent_data = round_data['agents'][role]
                                        end_state = agent_data['end_state']
                                        
                                        with cols[col_idx % 2]:
                                            # UseSubexpandercomponentsDisplayDetailedInfo
                                            with st.expander(f"{role_icons[role]} {role_names[role]}", expanded=True):
                                                # CreateSubColumnDisplayDetailedInfo
                                                sub_cols = st.columns(2)
                                                with sub_cols[0]:
                                                    st.metric("📦 Inventory", end_state['inventory'])
                                                    st.metric("📋 Order", agent_data['order_placed'])
                                                with sub_cols[1]:
                                                    st.metric("❌ Stockout", end_state['backorder'])
                                                    st.metric("💰 Cost", f"{end_state['round_cost']:.1f}")
                                        
                                        col_idx += 1
            
            # Run Simulation
            result = engine.run_simulation(
                progress_callback=progress_callback,
                realtime_callback=realtime_callback
            )
            
            # DisplaySimulation CompletedInfo
            if progress_container:
                with progress_container:
                    # CompletedProgressItem
                    overall_progress.progress(1.0)
                    progress_stats_container.success(f"🎯 Progress: {total_weeks}/{total_weeks} Round | ✅ Status: Completed | 📊 Completion Rate: 100.0%")
                
                overall_status.success(f"🎉 Simulation completed! Total rounds run: {result.total_rounds} Round")
                
                # Display finalStatistics Info
                final_stats_html = f'''
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
                    <h3 style="margin: 0 0 15px 0;">🏆 Simulation Completed - Statistics</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px; font-weight: bold;">{result.total_rounds}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Total Rounds</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px; font-weight: bold;">{len(result.round_history)}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Data Record</div>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                            <div style="font-size: 24px; font-weight: bold;">{result.total_cost:.0f}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Total Cost</div>
                        </div>
                    </div>
                </div>
                '''
                progress_container.markdown(final_stats_html, unsafe_allow_html=True)
                
                progress_container.markdown("### 🏁 Simulation Completed")
                progress_container.success(f"✅ Completed {result.total_rounds} Round，Total Cost: {result.total_cost:.2f}")
                
                # Display final Cost Analysis
                progress_container.markdown("#### 💰 Final Cost Analysis")
                role_names = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}
                for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                    cost = result.agent_costs[role]
                    percentage = (cost / result.total_cost * 100) if result.total_cost > 0 else 0
                    progress_container.text(f"   {role_names[role]}: {cost:.2f} ({percentage:.1f}%)")
            
            # AnalysisBullwhip Effect
            analyzer = BullwhipAnalyzer()
            bullwhip_metrics = analyzer.analyze_simulation_result(result)
            
            return result, bullwhip_metrics, None
            
        except Exception as e:
            if progress_container:
                progress_container.error(f"❌ SimulationFailed: {str(e)}")
            return None, None, str(e)
    
    def format_inventory_display(self, inventory_value):
        """Format Inventory Display，Negative Value displays as Stockout Quantity"""
        if inventory_value < 0:
            return f"Stockout {abs(inventory_value)} Units"
        elif inventory_value == 0:
            return "No Inventory"
        else:
            return f"{inventory_value} Units"
    
    def format_backorder_display(self, backorder_value):
        """Format Stockout Quantity Display"""
        if backorder_value > 0:
            return f"Stockout {backorder_value} Units"
        else:
            return "No Stockout"
    
    def plot_inventory_levels(self, result):
        """Plot inventory level chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Retailer Inventory', 'Wholesaler Inventory', 'Distributor Inventory', 'Manufacturer Inventory']
        )
        
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        colors = [self.role_colors[role] for role in roles]
        
        for i, (role, color) in enumerate(zip(roles, colors)):
            # fromround_historyinExtractInventoryData
            weeks = []
            inventory = []
            hover_text = []
            for round_data in result.round_history:
                if role in round_data['agents']:
                    weeks.append(round_data['round'])
                    # CalculateNetInventory:Inventory - StockoutQuantity（Negative Value Indicates Stockout）
                    inv_value = round_data['agents'][role]['end_state'].get('inventory', 0)
                    backorder_value = round_data['agents'][role]['end_state'].get('backorder', 0)
                    net_inventory = inv_value - backorder_value
                    inventory.append(net_inventory)
                    # AddHover textThisDisplayInventoryStatus
                    if net_inventory < 0:
                        hover_text.append(f"No.{round_data['round']}Round: Stockout {abs(net_inventory)} Units")
                    elif net_inventory == 0:
                        hover_text.append(f"No.{round_data['round']}Round: No Inventory")
                    else:
                        hover_text.append(f"No.{round_data['round']}Round: {net_inventory} Units")
            
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Based onInventoryValueSetUnifiedStatusColor:NegativeValueRed，ZeroValueYellow，PositiveValueGreen
            marker_colors = []
            line_colors = []
            for inv in inventory:
                if inv < 0:
                    marker_colors.append(self.inventory_status_colors['negative'])  # RedIndicatesStockout
                    line_colors.append(self.inventory_status_colors['negative'])
                elif inv == 0:
                    marker_colors.append(self.inventory_status_colors['zero'])      # YellowIndicatesNo Inventory
                    line_colors.append(self.inventory_status_colors['zero'])
                else:
                    marker_colors.append(self.inventory_status_colors['positive'])  # GreenIndicatesPositiveInventory
                    line_colors.append(self.inventory_status_colors['positive'])
            
            # UseUnifiedInventoryStatusColorPlotLineandMarker
            fig.add_trace(
                go.Scatter(
                    x=weeks, y=inventory,
                    mode='lines+markers',
                    name=f'{role}Inventory',
                    line=dict(color=color, width=2),  # KeepRoleoriginal colorForLine
                    marker=dict(size=8, color=marker_colors),  # UseStatusColorMarker
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Supply Chain Each Stage Inventory Level（Negative Value Indicates Stockout）',
            height=600,
            showlegend=False
        )
        
        return fig

    def plot_in_transit_levels(self, result):
        """Plot in-transit inventory total amount for each role"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Retailer In-Transit', 'Wholesaler In-Transit', 'Distributor In-Transit', 'Manufacturer In-Production']
        )

        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.role_colors[role] for role in roles]

        for i, (role, name, color) in enumerate(zip(roles, role_names, colors)):
            weeks = []
            in_transit = []
            hover_text = []
            for round_data in result.round_history:
                if role in round_data['agents']:
                    weeks.append(round_data['round'])
                    end_state = round_data['agents'][role]['end_state']
                    total_in_transit = end_state.get('total_in_transit')
                    if total_in_transit is None:
                        pipeline = end_state.get('shipment_pipeline', []) or []
                        total_in_transit = sum(pipeline)
                    in_transit.append(total_in_transit)
                    incoming = end_state.get('incoming_shipment', 0)
                    if role == 'manufacturer':
                        hover_text.append(f"No.{round_data['round']}Round: inProduction {total_in_transit} | Next PeriodCompleted {incoming}")
                    else:
                        hover_text.append(f"No.{round_data['round']}Round: inTransit {total_in_transit} | Next PeriodtoArrival {incoming}")

            row = i // 2 + 1
            col = i % 2 + 1

            series_name = f"{name}inProduction" if role == 'manufacturer' else f"{name}inTransit"
            fig.add_trace(
                go.Scatter(
                    x=weeks, y=in_transit,
                    mode='lines+markers',
                    name=series_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title='Supply Chain Each Stage In-Transit Inventory Total Amount',
            height=600,
            showlegend=False
        )

        return fig

    def plot_orders_and_demand(self, result):
        """Plot order and demand chart"""
        fig = go.Figure()
        
        # ExtractDemandData
        demand_history = []
        for round_data in result.round_history:
            demand_history.append(round_data['customer_demand'])
        
        weeks = list(range(1, len(demand_history) + 1))
        
        # Customer Demand
        fig.add_trace(go.Scatter(
            x=weeks, y=demand_history,
            mode='lines+markers',
            name='Customer Demand',
            line=dict(color=self.status_colors['demand'], width=3),
            marker=dict(size=8)
        ))
        
        # EachTierOrder
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.role_colors[role] for role in roles]
        
        for role, name, color in zip(roles, role_names, colors):
            # fromround_historyinExtractOrderData
            orders = []
            for round_data in result.round_history:
                if role in round_data['agents']:
                    orders.append(round_data['agents'][role].get('order_placed', 0))
            
            fig.add_trace(go.Scatter(
                x=weeks, y=orders,
                mode='lines+markers',
                name=f'{name}Order',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='Demand and Order Propagation',
            xaxis_title='Weeks',
            yaxis_title='Quantity',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_costs(self, result):
        """Plot cost analysis chart"""
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.role_colors[role] for role in roles]
        
        # Calculate Total Cost
        total_costs = []
        for role in roles:
            total_cost = result.agent_costs.get(role, 0)
            total_costs.append(total_cost)
        
        # CreateBarChart
        fig = go.Figure(data=[
            go.Bar(
                x=role_names,
                y=total_costs,
                marker_color=colors,
                text=[f'${cost:.0f}' for cost in total_costs],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Supply Chain Each Stage Total Cost',
            xaxis_title='Supply Chain Role',
            yaxis_title='Total Cost ($)',
            height=400
        )
        
        return fig

    def display_detailed_cost_analysis(self, result):
        """Display detailed cost analysis, including each participant's average cost per period and average total cost"""
        st.markdown("### 💰 Detailed Cost Analysis")
        
        # GetTotal Rounds
        total_rounds = len(getattr(result, 'round_history', []) or [])
        if total_rounds == 0:
            st.warning("Unable to get Simulation Round Count Info")
            return
        
        # CalculateEachRoleTotal Cost andAverageCost
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        
        # CreateTwo RowsLayout
        col1, col2 = st.columns(2)
        
        # In columns, display each participant's detailed cost info
        with col1:
            st.markdown("#### Each Participants Cost Details")
            
            # Calculate and display each participant's cost info
            for i, (role, name) in enumerate(zip(roles, role_names)):
                total_cost = result.agent_costs.get(role, 0)
                avg_cost_per_cycle = total_cost / total_rounds if total_rounds > 0 else 0
                
                # Usest.metricDisplayInfo
                st.metric(
                    label=f"{name}",
                    value=f"${total_cost:.2f}",
                    delta=f"AverageEachPeriod: ${avg_cost_per_cycle:.2f}"
                )
        
        # In Column 2 Display Overall Cost Info
        with col2:
            st.markdown("#### Overall Cost Overview")
            
            # Calculate Total Cost
            total_cost = result.total_cost
            avg_total_cost_per_cycle = total_cost / total_rounds if total_rounds > 0 else 0
            
            # Display Overall Cost Info
            st.metric(
                label="Supply Chain Total Cost",
                value=f"${total_cost:.2f}",
                delta=f"AverageEachPeriod: ${avg_total_cost_per_cycle:.2f}"
            )
            
            # Display Cost Breakdown Pie Chart
            cost_data = []
            cost_labels = []
            for role, name in zip(roles, role_names):
                cost = result.agent_costs.get(role, 0)
                if cost > 0:
                    cost_data.append(cost)
                    cost_labels.append(f"{name} (${cost:.2f})")
            
            if cost_data:
                fig_pie = go.Figure(data=[go.Pie(labels=cost_labels, values=cost_data)])
                fig_pie.update_layout(title="Cost Breakdown Distribution")
                st.plotly_chart(fig_pie, width='stretch')
        
        # Display Detailed Cost Table
        st.markdown("#### Cost Detailed Data Table")
        cost_table_data = []
        for role, name in zip(roles, role_names):
            total_cost = result.agent_costs.get(role, 0)
            avg_cost_per_cycle = total_cost / total_rounds if total_rounds > 0 else 0
            
            cost_table_data.append({
                "Participants": name,
                "Total Cost ($)": f"{total_cost:.2f}",
                "Average Each Period Cost ($)": f"{avg_cost_per_cycle:.2f}"
            })
        
        # Create DataFrame and Display
        import pandas as pd
        df = pd.DataFrame(cost_table_data)
        st.dataframe(df, width='stretch')
        
        # Add Download Button
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Download Cost Analysis Data",
            data=csv,
            file_name="Cost Analysis Details.csv",
            mime="text/csv"
        )
    
    def plot_total_cost_per_round(self, result):
        """Plot per-round total cost line chart (sum of all participants' costs)"""
        # GetTotal Rounds
        total_rounds = len(getattr(result, 'round_history', []) or [])
        if total_rounds == 0:
            st.warning("Unable to get Simulation Round Count Info")
            return go.Figure()
        
        # CalculatePer Round Total Cost（All ParticipantsCostSum）
        total_costs_per_round = []
        weeks = list(range(1, total_rounds + 1))
        
        # fromround_historyinExtractPer RoundEach ParticipantsCost andsumand
        for round_data in result.round_history:
            round_total_cost = 0
            for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                if role in round_data['agents']:
                    round_total_cost += round_data['agents'][role].get('round_cost', 0)
            total_costs_per_round.append(round_total_cost)
        
        # CreateLine Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weeks,
            y=total_costs_per_round,
            mode='lines+markers',
            name='Per Round Total Cost',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6)
        ))
        
        # Add Average Cost Line
        avg_cost = sum(total_costs_per_round) / len(total_costs_per_round) if total_costs_per_round else 0
        fig.add_hline(
            y=avg_cost,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"AverageCost: ${avg_cost:.2f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title='Per Round Total Cost（All ParticipantsCostSum）',
            xaxis_title='Week',
            yaxis_title='Total Cost ($)',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_bullwhip_effect(self, bullwhip_metrics):
        """Plot bullwhip effect analysis chart"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['CV', 'Amplification Ratio'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CV
        cv_roles = list(bullwhip_metrics.coefficient_of_variation.keys())
        cv_values = list(bullwhip_metrics.coefficient_of_variation.values())
        
        fig.add_trace(
            go.Bar(
                x=cv_roles,
                y=cv_values,
                name='CV',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Amplification Ratio
        amp_pairs = list(bullwhip_metrics.amplification_ratios.keys())
        amp_values = list(bullwhip_metrics.amplification_ratios.values())
        
        fig.add_trace(
            go.Bar(
                x=amp_pairs,
                y=amp_values,
                name='Amplification Ratio',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Bullwhip EffectAnalysis (WholeEffect: {bullwhip_metrics.overall_bullwhip_effect:.3f})',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_cost_trends(self, result):
        """Plot each participant's cost trend line chart (per round cost and cumulative cost)"""
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.role_colors[role] for role in roles]
        
        # CalculateTotal Rounds（PriorityUseround_historyLength）
        total_rounds = len(getattr(result, 'round_history', []) or [])
        if total_rounds == 0:
            # Ifround_historyNotavailable，thenBased oncost_historyMaxLengthestimate
            total_rounds = max([
                len(result.agent_states.get(role, {}).get('cost_history', []))
                for role in roles
            ] + [0])
        weeks = list(range(1, total_rounds + 1))
        
        # CreateUpDownTwo RowsSubChart:Up-Per Round Cost；Down-Cumulative Cost
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.12,
                            subplot_titles=("Per Round Cost (Each Participants)", "Cumulative Cost (Each Participants)"))
        
        for role, name, color in zip(roles, role_names, colors):
            costs = result.agent_states.get(role, {}).get('cost_history', [])
            # padLengthtototal_rounds
            if len(costs) < total_rounds:
                costs = costs + [0] * (total_rounds - len(costs))
            else:
                costs = costs[:total_rounds]
            cum_costs = list(np.cumsum(costs))
            
            # Per Round Cost
            fig.add_trace(
                go.Scatter(x=weeks, y=costs, name=f"{name}Per Round Cost",
                           line=dict(color=color, width=2)),
                row=1, col=1
            )
            # Cumulative Cost
            fig.add_trace(
                go.Scatter(x=weeks, y=cum_costs, name=f"{name}Cumulative Cost",
                           line=dict(color=color, width=2, dash='dot')),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            hovermode='x unified',
            xaxis_title='Week',
            xaxis2_title='Week',
            yaxis_title='Per Round Cost ($)',
            yaxis2_title='Cumulative Cost ($)'
        )
        return fig
    
    def plot_bullwhip_timeseries(self, result, window: int = 5):
        """Plot each participant's order quantity and rolling CV line chart"""
        roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer']
        role_names = ['Retailer', 'Wholesaler', 'Distributor', 'Manufacturer']
        colors = [self.role_colors[role] for role in roles]
        
        # CalculateTotal Rounds
        total_rounds = len(getattr(result, 'round_history', []) or [])
        if total_rounds == 0:
            total_rounds = max([
                len(result.agent_states.get(role, {}).get('orders_history', []))
                for role in roles
            ] + [0])
        weeks = list(range(1, total_rounds + 1))
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.12,
                            subplot_titles=("Order Quantity Time Series (Each Participant)", f"RollingCVCV（window={window}）"))
        
        for role, name, color in zip(roles, role_names, colors):
            orders = result.agent_states.get(role, {}).get('orders_history', [])
            if len(orders) < total_rounds:
                orders = orders + [0] * (total_rounds - len(orders))
            else:
                orders = orders[:total_rounds]
            
            # Top:OrderTime Series
            fig.add_trace(
                go.Scatter(x=weeks, y=orders, name=f"{name}Order",
                           line=dict(color=color, width=2)),
                row=1, col=1
            )
            
            # Bottom:RollingCV（std/mean），meanfor0WhenSet0
            s = pd.Series(orders, dtype='float')
            rolling_std = s.rolling(window=window).std()
            rolling_mean = s.rolling(window=window).mean()
            cv = rolling_std / rolling_mean
            cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cv_values = cv.tolist()
            
            fig.add_trace(
                go.Scatter(x=weeks, y=cv_values, name=f"{name}CV",
                           line=dict(color=color, width=2, dash='dot')),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            hovermode='x unified',
            xaxis_title='Week',
            xaxis2_title='Week',
            yaxis_title='Order Quantity',
            yaxis2_title='CV（Rolling）'
        )
        return fig
    
    def display_summary_metrics(self, result, bullwhip_metrics):
        """Display Summary Metric"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate Total Cost
        total_cost = result.total_cost
        
        with col1:
            st.metric("Total Cost", f"${total_cost:.0f}")
        
        with col2:
            st.metric("Bullwhip Effect", f"{bullwhip_metrics.overall_bullwhip_effect:.3f}")
        
        with col3:
            # fromround_historyinCalculateAverageInventory
            role_inventories = {'retailer': [], 'wholesaler': [], 'distributor': [], 'manufacturer': []}
            for round_data in result.round_history:
                for role in role_inventories.keys():
                    if role in round_data['agents']:
                        inventory = round_data['agents'][role]['end_state'].get('inventory', 0)
                        role_inventories[role].append(inventory)
            
            avg_inventory = np.mean([np.mean(inventories) for inventories in role_inventories.values() if inventories])
            st.metric("AverageInventory", f"{avg_inventory:.1f}")
        
        with col4:
            # fromround_historyinCalculateService Level
            total_shortage = 0
            total_demand = 0
            for round_data in result.round_history:
                total_demand += round_data.get('customer_demand', 0)
                if 'retailer' in round_data['agents']:
                    shortage = round_data['agents']['retailer']['end_state'].get('backorder', 0)
                    total_shortage += shortage
            
            service_level = 1.0 - (total_shortage / total_demand) if total_demand > 0 else 1.0
            st.metric("Service Level", f"{service_level:.1%}")
        
        # Add Order Limit Statistics
        self._display_order_limits_stats(result)
    
    def _display_order_limits_stats(self, result):
        """Display order limit statistics info"""
        st.markdown("### 📊 Order Limit Statistics")
        
        # Check whether order limit is enabled
        order_limits_enabled = False
        roles_with_limits = {}
        
        # Get order limit info from config
        if hasattr(result, 'config'):
            config = result.config
            for role in ['retailer', 'wholesaler', 'distributor', 'manufacturer']:
                role_config = getattr(config, role, None)
                if role_config and hasattr(role_config, 'min_order_quantity') and hasattr(role_config, 'max_order_quantity'):
                    min_qty = role_config.min_order_quantity
                    max_qty = role_config.max_order_quantity
                    if (min_qty is not None and min_qty > 0) or (max_qty is not None and max_qty < 100):  # Check whether non-default values are set
                        order_limits_enabled = True
                        roles_with_limits[role] = {'min': min_qty, 'max': max_qty}
        
        if not order_limits_enabled:
            st.info("📝 This simulation does not have Order Limit Feature enabled。")
            return
        
        # Statistics: Order Limit Application Status
        limit_violations = {role: {'below_min': 0, 'above_max': 0, 'total_orders': 0} for role in roles_with_limits.keys()}
        
        for round_data in result.round_history:
            for role, limits in roles_with_limits.items():
                if role in round_data['agents']:
                    agent_data = round_data['agents'][role]
                    order_quantity = agent_data.get('order_placed', 0)  # FixFieldName:Useorder_placedrather thanorder_quantity
                    
                    limit_violations[role]['total_orders'] += 1
                    
                    if order_quantity < limits['min']:
                        limit_violations[role]['below_min'] += 1
                    elif order_quantity > limits['max']:
                        limit_violations[role]['above_max'] += 1
        
        # Display Statistics Result
        cols = st.columns(len(roles_with_limits))
        role_names = {'retailer': 'Retailer', 'wholesaler': 'Wholesaler', 'distributor': 'Distributor', 'manufacturer': 'Manufacturer'}
        
        for i, (role, limits) in enumerate(roles_with_limits.items()):
            with cols[i]:
                violations = limit_violations[role]
                total_violations = violations['below_min'] + violations['above_max']
                compliance_rate = 1.0 - (total_violations / violations['total_orders']) if violations['total_orders'] > 0 else 1.0
                
                st.metric(
                    f"{role_names[role]}Compliance Rate",
                    f"{compliance_rate:.1%}",
                    delta=f"Limit: {limits['min']}-{limits['max']}"
                )
                
                if total_violations > 0:
                    st.caption(f"Violation: {total_violations}times")

    def display_decision_statistics(self, result):
        """DisplayFourParticipantsOrder DecisionStatisticsChart"""
        st.markdown("### 📊 Participants Order Decision Statistics Analysis")
        st.markdown("""This analysis displays four participants' order decision mode、Compliance Rate and Decision Probability Distribution，Helps understand decision behavior characteristics of each participant。""")
        
        # ExtractOrder DecisionData
        roles = {
            'retailer': '🏪 Retailer',
            'wholesaler': '🏢 Wholesaler', 
            'distributor': '🚚 Distributor',
            'manufacturer': '🏭 Manufacturer'
        }
        
        # Calculate Decision Statistics Data
        decision_stats = {}
        for role_key, role_name in roles.items():
            orders = []
            demands = []
            violations = 0
            total_decisions = 0
            
            for round_data in result.round_history:
                if role_key in round_data['agents']:
                    agent_data = round_data['agents'][role_key]
                    order = agent_data.get('order_placed', 0)
                    demand = agent_data.get('demand_received', 0)
                    
                    orders.append(order)
                    demands.append(demand)
                    total_decisions += 1
                    
                    # CheckwhetherViolationOrderLimit（AssumingMaxOrder QuantityforDemand3times）
                    max_allowed = max(demand * 3, 100)  # Minallow100
                    if order > max_allowed:
                        violations += 1
            
            # CalculateStatisticsMetric
            if orders:
                decision_stats[role_key] = {
                    'name': role_name,
                    'orders': orders,
                    'demands': demands,
                    'avg_order': sum(orders) / len(orders),
                    'max_order': max(orders),
                    'min_order': min(orders),
                    'compliance_rate': ((total_decisions - violations) / total_decisions * 100) if total_decisions > 0 else 100,
                    'violations': violations,
                    'total_decisions': total_decisions,
                    'order_variance': sum((x - sum(orders)/len(orders))**2 for x in orders) / len(orders) if len(orders) > 1 else 0
                }
        
        # Display Compliance Rate Overview
        st.markdown("#### 🎯 Compliance Rate Overview")
        cols = st.columns(4)
        for i, (role_key, stats) in enumerate(decision_stats.items()):
            with cols[i]:
                st.metric(
                    label=stats['name'],
                    value=f"{stats['compliance_rate']:.1f}%",
                    delta=f"{stats['violations']} Violation" if stats['violations'] > 0 else "Fully Compliant"
                )
        
        # Create Four Decision Statistics Charts
        st.markdown("#### 📈 Order DecisionProbability Distribution")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        # Create 2x2 Sub Chart Layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[stats['name'] for stats in decision_stats.values()],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (role_key, stats) in enumerate(decision_stats.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            color = colors[i]
            
            orders = stats['orders']
            demands = stats['demands']
            
            # Calculate Order Quantity Probability Distribution
            if orders:
                # Create Order Quantity Histogram Chart Data
                order_bins = np.linspace(min(orders), max(orders), 20)
                hist, bin_edges = np.histogram(orders, bins=order_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                probabilities = hist / sum(hist) * 100  # Convert topercentageRatio
                
                # Add Probability Distribution Bar Chart
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=probabilities,
                        name=f'{stats["name"]}Probability',
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add Average Order Quantity Line
                fig.add_vline(
                    x=stats['avg_order'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {stats['avg_order']:.1f}",
                    row=row, col=col
                )
        
        # UpdateLayout
        fig.update_layout(
            height=800,
            title_text="FourParticipantsOrder DecisionProbability DistributionStatistics",
            showlegend=False
        )
        
        # Update x-axis and y-axis labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Order Quantity", row=i, col=j)
                fig.update_yaxes(title_text="Probability (%)", row=i, col=j)
        
        st.plotly_chart(fig, width='stretch')
        
        # Display Detailed Statistics Table
        st.markdown("#### 📋 Detailed Decision Statistics")
        stats_df = []
        for role_key, stats in decision_stats.items():
            stats_df.append({
                'Participants': stats['name'],
                'AverageOrder Quantity': f"{stats['avg_order']:.2f}",
                'MaxOrder Quantity': stats['max_order'],
                'MinOrder Quantity': stats['min_order'],
                'OrderVariance': f"{stats['order_variance']:.2f}",
                'Compliance Rate': f"{stats['compliance_rate']:.1f}%",
                'ViolationCount': stats['violations'],
                'TotalDecisionCount': stats['total_decisions']
            })
        
        import pandas as pd
        df = pd.DataFrame(stats_df)
        st.dataframe(df, width='stretch')
        
        # DecisionRowforAnalysis
        st.markdown("#### 🔍 Decision Row for Analysis")
        
        for role_key, stats in decision_stats.items():
            with st.expander(f"📊 {stats['name']} DecisionAnalysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Decision Characteristics:**")
                    if stats['order_variance'] > 1000:
                        st.write("• Decision fluctuation is relatively large, possible overreaction")
                    elif stats['order_variance'] < 100:
                        st.write("• Decision is relatively stable, strategy is relatively conservative")
                    else:
                        st.write("• Decision fluctuation is moderate, strategy is relatively balanced")
                    
                    if stats['compliance_rate'] < 90:
                        st.write("• ⚠️ Compliance rate is low, need to optimize decision strategy")
                    elif stats['compliance_rate'] < 95:
                        st.write("• ⚡ Compliance rate is good, with occasional violations")
                    else:
                        st.write("• ✅ Compliance rate is excellent, decision strategy is reasonable")
                
                with col2:
                    st.markdown("**Improvement Suggestions:**")
                    if stats['violations'] > 0:
                        st.write(f"• Reduce violation decisions, currently {stats['violations']} violations")
                    if stats['order_variance'] > 1000:
                        st.write("• Consider a smoother order strategy")
                    if stats['avg_order'] > sum(stats['demands'])/len(stats['demands']) * 2:
                        st.write("• Order quantity is biased high, may lead to inventory backlog")
                    if stats['avg_order'] < sum(stats['demands'])/len(stats['demands']) * 0.8:
                        st.write("• Order quantity is low, may lead to stockout risk")
        
        # AddDecisionDetailsDisplayPanel
        st.markdown("#### 📋 Decision Details and Reason Explanations by Period")
        st.markdown("Displays each participant's specific order decisions and LLM reasoning explanation for each period.")
        
        # CreateDecisionDetailsData
        decision_details = []
        for round_idx, round_data in enumerate(result.round_history):
            round_num = round_idx + 1
            for role_key, role_name in roles.items():
                if role_key in round_data['agents']:
                    agent_data = round_data['agents'][role_key]
                    order_placed = agent_data.get('order_placed', 0)
                    demand_received = agent_data.get('demand_received', 0)
                    inventory = agent_data.get('inventory', 0)
                    backlog = agent_data.get('backorder', 0)

                    # Get decision reason from round_data (saved by game_engine)
                    decision_reason = agent_data.get('decision_reason', '')
                    if not decision_reason:
                        decision_reason = agent_data.get('decision_explanation', 'Not provided')

                    decision_details.append({
                        'Period': round_num,
                        'Participants': role_name,
                        'Order Quantity': order_placed,
                        'Demand Quantity': demand_received,
                        'Inventory': inventory,
                        'Stockout': backlog,
                        'Decision Reason': decision_reason
                    })
        
        # DisplayDecisionDetailsTable
        if decision_details:
            import pandas as pd
            details_df = pd.DataFrame(decision_details)
            
            # AddFilterOption
            col1, col2 = st.columns(2)
            with col1:
                selected_participants = st.multiselect(
                    "Select Participants",
                    options=list(roles.values()),
                    default=list(roles.values()),
                    key="decision_details_participants"
                )
            
            with col2:
                max_rounds = len(result.round_history)
                selected_rounds = st.slider(
                    "Select Period Range",
                    min_value=1,
                    max_value=max_rounds,
                    value=(1, min(10, max_rounds)),
                    key="decision_details_rounds"
                )
            
            # FilterData
            filtered_df = details_df[
                (details_df['Participants'].isin(selected_participants)) &
                (details_df['Period'] >= selected_rounds[0]) &
                (details_df['Period'] <= selected_rounds[1])
            ]
            
            # DisplayFilterafterData
            st.dataframe(
                filtered_df,
                width='stretch',
                column_config={
                    'Period': st.column_config.NumberColumn('Period', width='small'),
                    'Participants': st.column_config.TextColumn('Participants', width='medium'),
                    'Order Quantity': st.column_config.NumberColumn('Order Quantity', width='small'),
                    'Demand Quantity': st.column_config.NumberColumn('Demand Quantity', width='small'),
                    'Inventory': st.column_config.NumberColumn('Inventory', width='small'),
                    'Stockout': st.column_config.NumberColumn('Stockout', width='small'),
                    'Decision Reason': st.column_config.TextColumn('Decision Reason', width='large')
                }
            )
            
            # AddExpandableDetailedViewChart
            st.markdown("#### 🔍 Detailed Decision Analysis")
            for role_key, role_name in roles.items():
                if role_name in selected_participants:
                    with st.expander(f"📊 {role_name} DetailedDecisionHistory", expanded=False):
                        role_decisions = filtered_df[filtered_df['Participants'] == role_name]
                        
                        if not role_decisions.empty:
                            for _, row in role_decisions.iterrows():
                                st.markdown(f"**Period {row['Period']}:**")
                                col1, col2, col3 = st.columns([1, 1, 2])
                                
                                with col1:
                                    st.metric("Order Quantity", row['Order Quantity'])
                                    st.metric("Demand Quantity", row['Demand Quantity'])
                                
                                with col2:
                                    st.metric("Inventory", row['Inventory'])
                                    st.metric("Stockout", row['Stockout'])
                                
                                with col3:
                                    st.markdown("**Decision Reason:**")
                                    st.write(row['Decision Reason'])
                                
                                st.divider()
                        else:
                            st.info("No decision data for the selected participants in the given period range")
        else:
            st.warning("Not foundDecisionDetailsData，PossibleisbecauseforSimulation ResultsinmissingDecision ReasonInfo。")

    def display_coordinator_analysis(self, result):
        """Coordinator analysis — disabled"""
        st.info("Coordinator feature has been disabled.")
        return

    def save_all_data(self, result, bullwhip_metrics, ui_config):
        """Save all simulation data to a ZIP file"""
        try:
            # CreateTempDirectory
            temp_dir = tempfile.mkdtemp()
            
            # 1. SaveSimulation ResultsJSON
            result_dict = {
                'config': ui_config,
                'round_history': result.round_history,
                'total_cost': result.total_cost,

            }
            with open(os.path.join(temp_dir, 'simulation_result.json'), 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            # 2. SaveBullwhip EffectAnalysis
            with open(os.path.join(temp_dir, 'bullwhip_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(bullwhip_metrics, f, indent=2, ensure_ascii=False, default=str)
            
            # 3. SaveEachRoleOperationsDataCSV
            roles = {
                'retailer': 'Retailer',
                'wholesaler': 'Wholesaler', 
                'distributor': 'Distributor',
                'manufacturer': 'Manufacturer'
            }
            
            for role_key, role_name in roles.items():
                role_data = pd.DataFrame([
                    {
                        "Round": round_data['round'],
                        "Received Demand": round_data['agents'][role_key].get('demand_received', 0),
                        "DownSingleDecision": round_data['agents'][role_key].get('order_placed', 0),
                        "Period StartInventory": self.format_inventory_display(round_data['agents'][role_key]['start_state'].get('inventory', 0)),
                        "Period EndInventory": self.format_inventory_display(round_data['agents'][role_key]['end_state'].get('inventory', 0)),
                        "StockoutQuantity": self.format_backorder_display(round_data['agents'][role_key]['end_state'].get('backorder', 0)),
                        "Holding Cost": round_data['agents'][role_key]['end_state'].get('holding_cost', 0),
                        "Stockout Cost": round_data['agents'][role_key]['end_state'].get('shortage_cost', 0),
                        "Total Cost": round_data['agents'][role_key].get('round_cost', 0)
                    }
                    for round_data in result.round_history
                    if role_key in round_data['agents']
                ])
                role_data.to_csv(os.path.join(temp_dir, f'{role_name}_OperationsData.csv'), 
                               index=False, encoding='utf-8-sig')
            
            # 4. SaveDemandData
            demand_data = pd.DataFrame([
                {
                    "Round": int(round_data['round']),
                    "Market Demand": int(round_data.get('customer_demand', 0)),
                    "RetailerReceived Demand": int(round_data['agents']['retailer'].get('demand_received', 0)) if 'retailer' in round_data['agents'] else 0,
                    "WholesalerReceived Demand": int(round_data['agents']['wholesaler'].get('demand_received', 0)) if 'wholesaler' in round_data['agents'] else 0,
                    "DistributorReceived Demand": int(round_data['agents']['distributor'].get('demand_received', 0)) if 'distributor' in round_data['agents'] else 0,
                    "ManufacturerReceived Demand": int(round_data['agents']['manufacturer'].get('demand_received', 0)) if 'manufacturer' in round_data['agents'] else 0
                }
                for round_data in result.round_history
            ])
            demand_data.to_csv(os.path.join(temp_dir, 'DemandData.csv'), 
                             index=False, encoding='utf-8-sig')
            
            # 5. SaveChartData
            # InventoryData
            inventory_data = []
            for round_data in result.round_history:
                for role_key, role_name in roles.items():
                    if role_key in round_data['agents']:
                        inventory_data.append({
                            'Round': round_data['round'],
                            'Role': role_name,
                            'Inventory': self.format_inventory_display(round_data['agents'][role_key]['end_state'].get('inventory', 0)),
                            'Stockout': self.format_backorder_display(round_data['agents'][role_key]['end_state'].get('backorder', 0))
                        })
            pd.DataFrame(inventory_data).to_csv(os.path.join(temp_dir, 'InventoryData.csv'), 
                                              index=False, encoding='utf-8-sig')
            
            # 6. Coordinator Analysis — disabled
            # 7. CreateZIPFile
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            zip_filename = f'BeerGame Simulation Data_{timestamp}.zip'
            zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            # CleanupTempDirectory
            shutil.rmtree(temp_dir)
            
            return zip_path, zip_filename
            
        except Exception as e:
            st.error(f"Save data failed: {e}")
            return None, None
    
    def run_app(self):
        """Run Streamlit app"""
        st.title("🍺 LLM Beer Game Simulation System")
        st.markdown("---")
        
        # Render sidebar
        ui_config = self.render_sidebar()
        
        # MainInterface — full-width controls + config preview
        st.markdown("---")

        # === Simulation Controls ===
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            run_button = st.button("🚀 Start Simulation", type="primary", width='stretch')

        # Feature toggles — hardcoded off, removed from UI per user request
        enable_realtime_3d = False
        show_realtime_status = False

        # Demand forecasting deduction mode (add-on checkbox)
        enable_demand_forecasting = st.checkbox(
            "🔮 Enable Supply Chain Demand Forecasting Deduction",
            value=st.session_state.get("enable_demand_forecasting", False),
            help="When checked, appends demand forecasting deduction rules to the end of original prompts without modifying any existing prompt text"
        )
        st.session_state.enable_demand_forecasting = enable_demand_forecasting

        # === Config Preview ===
        st.markdown("---")
        st.markdown("### 📋 Config Preview")

        demand_pattern = ui_config["demand_pattern"]

        # --- Top-row summary cards ---
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)

        demand_label = demand_pattern.replace(" Demand", "")
        card_col1.markdown(f"""<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:16px 12px;border-radius:10px;text-align:center;">
            <div style="font-size:12px;opacity:0.85;margin-bottom:4px;">📊 Demand Type</div>
            <div style="font-size:18px;font-weight:bold;">{demand_label}</div></div>""", unsafe_allow_html=True)

        card_col2.markdown(f"""<div style="background:linear-gradient(135deg,#11998e,#38ef7d);color:#fff;padding:16px 12px;border-radius:10px;text-align:center;">
            <div style="font-size:12px;opacity:0.85;margin-bottom:4px;">🔢 Rounds</div>
            <div style="font-size:18px;font-weight:bold;">{ui_config['num_rounds']}</div></div>""", unsafe_allow_html=True)

        card_col3.markdown(f"""<div style="background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;padding:16px 12px;border-radius:10px;text-align:center;">
            <div style="font-size:12px;opacity:0.85;margin-bottom:4px;">🎲 Random Seed</div>
            <div style="font-size:18px;font-weight:bold;">{ui_config['random_seed']}</div></div>""", unsafe_allow_html=True)

        info_sharing = "Yes" if ui_config.get('enable_info_sharing', False) else "No"
        card_col4.markdown(f"""<div style="background:linear-gradient(135deg,#4facfe,#00f2fe);color:#fff;padding:16px 12px;border-radius:10px;text-align:center;">
            <div style="font-size:12px;opacity:0.85;margin-bottom:4px;">🔄 Info Sharing</div>
            <div style="font-size:18px;font-weight:bold;">{info_sharing}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Demand parameters card ---
        with st.container():
            st.markdown("#### 📊 Demand Parameters")
            dcol1, dcol2, dcol3 = st.columns(3)

            if demand_pattern == "Step Demand":
                with dcol1:
                    st.metric("Initial Demand", ui_config.get("initial_demand", 4))
                with dcol2:
                    st.metric("Step Demand", ui_config.get("step_demand", 8))
                with dcol3:
                    st.metric("Step Week", ui_config.get("step_week", 5))
                st.caption(f"Weeks 1–{ui_config.get('step_week', 5)-1}: demand = {ui_config.get('initial_demand', 4)} | "
                          f"Weeks {ui_config.get('step_week', 5)}+: demand = {ui_config.get('step_demand', 8)}")
            elif demand_pattern == "Constant Demand":
                with dcol1:
                    st.metric("Constant Demand", ui_config.get("constant_demand", 4))
            elif demand_pattern == "Random Demand":
                min_d = ui_config.get("random_min", 1)
                max_d = ui_config.get("random_max", 10)
                with dcol1:
                    st.metric("Min Demand", min_d)
                with dcol2:
                    st.metric("Max Demand", max_d)
                with dcol3:
                    st.metric("Range", f"[{min_d}, {max_d}]")
                st.caption(f"Uniform distribution — each value in [{min_d}, {max_d}] equally likely")
            elif demand_pattern == "Seasonal Demand":
                with dcol1:
                    st.metric("Base Demand", ui_config.get("base_demand", 25))
                with dcol2:
                    st.metric("Amplitude", ui_config.get("amplitude", 10))
                with dcol3:
                    st.metric("Period", ui_config.get("period", 12))
            elif demand_pattern == "Normal Distribution":
                with dcol1:
                    st.metric("Mean", ui_config.get("normal_mean", 10.0))
                with dcol2:
                    st.metric("Std Dev", ui_config.get("normal_std", 2.0))
            elif demand_pattern == "Poisson Distribution":
                with dcol1:
                    st.metric("Lambda", ui_config.get("poisson_lambda", 8.0))
            else:
                with dcol1:
                    st.metric("Type", demand_pattern)

        # Agent config summary
        with st.expander("🏭 Supply Chain Agent Configuration", expanded=False):
            agent_cols = st.columns(4)
            roles = [
                ("retailer", "🏪 Retailer"),
                ("wholesaler", "🏢 Wholesaler"),
                ("distributor", "🚚 Distributor"),
                ("manufacturer", "🏭 Manufacturer"),
            ]
            for i, (role_key, role_label) in enumerate(roles):
                with agent_cols[i]:
                    inv = ui_config.get(f"{role_key}_initial_inventory", 15)
                    bl = ui_config.get(f"{role_key}_backlog", 0)
                    st.markdown(f"**{role_label}**")
                    st.caption(f"Initial Inventory: {inv}")
                    st.caption(f"Initial Backorder: {bl}")

        with st.expander("⚙️ View Full Config (JSON)", expanded=False):
            st.json(ui_config)
        
        # Run Simulation
        if run_button:
            # CreateReal-time3DDisplayContainer
            realtime_3d_container = None
            realtime_3d_viz = None
            
            if enable_realtime_3d:
                from llm_beer_game.visualization.realtime_3d import RealtimeSupplyChain3D
                realtime_3d_viz = RealtimeSupplyChain3D()
                
                st.markdown("### 🏢 Real-time 3D Supply Chain Visualization")
                
                # Create full-width container for 3D display
                realtime_3d_container = st.empty()
                
                # Initialize 3D display container
                realtime_3d_viz.initialize_display(realtime_3d_container)
            
            # Simulation Progress Display Container (only created when running)
            progress_container = None
            
            with st.spinner("Running simulation..."):
                # CreateSimulationProgressDisplayContainer
                # st.markdown("---")
                st.markdown("## 🎮 Simulation Progress")
                progress_container = st.container()
                
                game_config = self.create_game_config(ui_config)
                
                # Define real-time callback function
                def realtime_callback(data):
                    if realtime_3d_viz and realtime_3d_container:
                        realtime_3d_viz.update_round_data(data)
                        # NotneedneedReNewre-renderHTML，update_round_dataalreadyalreadyHandleDoneUpdate
                
                result, bullwhip_metrics, error = self.run_simulation(
                    game_config, 
                    realtime_callback if enable_realtime_3d else None,
                    progress_container=progress_container,
                    show_realtime_status=show_realtime_status
                )
                
                if error:
                    st.error(f"Simulation run failed: {error}")
                    progress_container.empty()  # Clear progress display
                else:
                    st.session_state.result = result
                    st.session_state.bullwhip_metrics = bullwhip_metrics
                    st.success("Simulation run completed!")

                    # Auto-collapse sidebar after simulation to maximize results view
                    st.components.v1.html("""
                    <script>
                    (function() {
                        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                        if (sidebar) {
                            var btn = sidebar.querySelector('button[data-testid="stSidebarCollapsedControl"]');
                            if (btn) btn.click();
                        }
                    })();
                    </script>
                    """, height=0)
        
        # DisplayResult
        if st.session_state.get('result'):
            st.markdown("---")
            
            # Result Title and Save Button
            result_col1, result_col2 = st.columns([3, 1])
            with result_col1:
                st.markdown("## 📈 Simulation Results")
            with result_col2:
                if st.button("💾 Save All Data", type="primary", help="Package all simulation data for download as ZIP file"):
                    with st.spinner("Packaging data..."):
                        zip_path, zip_filename = self.save_all_data(
                            st.session_state.result, 
                            st.session_state.bullwhip_metrics, 
                            ui_config
                        )
                        
                        if zip_path and zip_filename:
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {zip_filename}",
                                    data=f.read(),
                                    file_name=zip_filename,
                                    mime="application/zip",
                                    type="secondary"
                                )
                            st.success("Data package completed! Click the button above to download.")
                        else:
                            st.error("Data package failed, please retry.")
            
            # Summary Metric
            self.display_summary_metrics(st.session_state.result, st.session_state.bullwhip_metrics)
            
            # ChartDisplay
            tabs = ["🏢 3D Supply Chain", "📊 Inventory & Orders", "🚚 Shipment & Pipeline", "💰 Cost Analysis", "🔄 Bullwhip & Decision Stats"]

            # If debug features enabled, add debug tab
            if ui_config.get('enable_prompt_debug', False):
                tabs.append("🔧 Prompt Adjust / Debug")

            tab_objects = st.tabs(tabs)
            tab1, tab2, tab3, tab4, tab5 = tab_objects[:5]

            # Debug tab
            debug_tab = tab_objects[5] if len(tab_objects) > 5 else None

            with tab1:
                st.markdown("### 🏢 3D Supply Chain Dynamic Visualization")
                st.markdown("""This is an interactive 3D supply chain visualization that displays the dynamic process of order flow, inventory changes, and in-transit goods arrival between each role.

**Feature Description:**
- 🎮 **Play Control**: Use Play button to auto-play entire simulation process
- 🎚️ **Round Slider**: Manual Select to View Specific Round Status
- ⚡ **Play Speed**: Adjust animation playback speed (1x-10x)
- 🔄 **Reset Perspective**: Reset to Default 3D Perspective
- 🖱️ **Interaction**: Mouse drag to rotate view, scroll to zoom

**Visualization Elements:**
- 📦 **Supply Chain Nodes**: Different colors and sizes represent inventory status of each role
- ➡️ **Order Flow**: Green arrows show order direction and quantity
- 📊 **Real-time Info**: Top-right corner displays current round key metrics
- 🏷️ **Chart Legend**: Bottom-left corner displays each role color markers
                """)

                try:
                    visualizer_3d = SupplyChain3DVisualizer()
                    visualizer_3d.render_in_streamlit(st.session_state.result, width=1200, height=700)

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save 3D Visualization HTML", key="save_3d_viz"):
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = f"supply_chain_3d_{timestamp}.html"
                            visualizer_3d.save_html(st.session_state.result, output_path)
                            st.success(f"✅ 3D visualization saved to: {output_path}")

                    with col2:
                        st.info("💡 Tip: Saved HTML files can be opened standalone in Browser, supporting full interactive features.")

                except Exception as e:
                    st.error(f"❌ 3D visualization load failed: {str(e)}")
                    st.markdown("""**Possible Solutions:**
                    1. Ensure network connection is normal (need to load Three.js library)
                    2. Try refreshing the page
                    3. Check if Browser supports WebGL
                    """)

            with tab2:
                st.markdown("### 📊 Inventory Level")
                fig = self.plot_inventory_levels(st.session_state.result)
                st.plotly_chart(fig, width='stretch')

                st.markdown("---")
                st.markdown("### 📈 Order & Demand")
                fig = self.plot_orders_and_demand(st.session_state.result)
                st.plotly_chart(fig, width='stretch')

            with tab3:
                st.markdown("### 🚚 Shipment Quantity by Participant")
                st.markdown("""This chart displays each stage's shipment quantity change trend, helping analyze each participant's shipment arrival capability and response mode.""")

                visualizer = BeerGameVisualizer()
                fig = visualizer.plot_shipments_interactive(st.session_state.result)
                st.plotly_chart(fig, width='stretch')

                st.markdown("---")
                st.markdown("### 🚢 In-Transit Inventory (Each Participant)")
                st.markdown("This chart displays each participant's in-transit inventory total amount per round, combined with next-period arrival info to help evaluate replenishment rhythm and supply chain response.")
                fig_intransit = self.plot_in_transit_levels(st.session_state.result)
                st.plotly_chart(fig_intransit, width='stretch')

            with tab4:
                fig = self.plot_costs(st.session_state.result)
                st.plotly_chart(fig, width='stretch')

                self.display_detailed_cost_analysis(st.session_state.result)

                st.markdown("---")
                st.markdown("### 📊 Per Round Total Cost (Sum of All Participants' Costs)")
                fig_total_per_round = self.plot_total_cost_per_round(st.session_state.result)
                st.plotly_chart(fig_total_per_round, width='stretch')

                st.markdown("---")
                st.markdown("### 📈 Each Participant Cost Trend (Per Round and Cumulative)")
                fig_trends = self.plot_cost_trends(st.session_state.result)
                st.plotly_chart(fig_trends, width='stretch')

            with tab5:
                st.markdown("### 🔄 Bullwhip Effect")
                fig = self.plot_bullwhip_effect(st.session_state.bullwhip_metrics)
                st.plotly_chart(fig, width='stretch')

                st.markdown("---")
                st.markdown("### 🌊 Bullwhip Effect Detailed Trend (Order and Rolling CV)")
                fig_bw_ts = self.plot_bullwhip_timeseries(st.session_state.result, window=5)
                st.plotly_chart(fig_bw_ts, width='stretch')

                st.markdown("---")
                self.display_decision_statistics(st.session_state.result)
            
            # Debug Panel
            if debug_tab is not None:
                with debug_tab:
                    self.display_prompt_debug_panel(st.session_state.result, ui_config)
            
            # DetailedData
            with st.expander("ViewDetailedData", expanded=False):
                # DemandInfoDisplay
                st.markdown("### 📊 Full Period Demand Info")
                demand_data = pd.DataFrame([
                    {
                        "Round": int(round_data['round']),
                        "Market Demand": int(round_data.get('customer_demand', 0)),  # FixPositiveFieldName
                        "RetailerReceived Demand": int(round_data['agents']['retailer'].get('demand_received', 0)) if 'retailer' in round_data['agents'] else 0,
                        "WholesalerReceived Demand": int(round_data['agents']['wholesaler'].get('demand_received', 0)) if 'wholesaler' in round_data['agents'] else 0,
                        "DistributorReceived Demand": int(round_data['agents']['distributor'].get('demand_received', 0)) if 'distributor' in round_data['agents'] else 0,
                        "ManufacturerReceived Demand": int(round_data['agents']['manufacturer'].get('demand_received', 0)) if 'manufacturer' in round_data['agents'] else 0
                    }
                    for round_data in st.session_state.result.round_history
                ])
                # EnsureAllNumberValueColumnallisNumberValueType
                numeric_columns = ["Round", "Market Demand", "RetailerReceived Demand", "WholesalerReceived Demand", "DistributorReceived Demand", "ManufacturerReceived Demand"]
                for col in numeric_columns:
                    demand_data[col] = pd.to_numeric(demand_data[col], errors='coerce').fillna(0).astype(int)
                
                st.dataframe(demand_data, width='stretch')
                
                # EachRoleDetailedOperationsData
                roles = {
                    'retailer': 'Retailer',
                    'wholesaler': 'Wholesaler', 
                    'distributor': 'Distributor',
                    'manufacturer': 'Manufacturer'
                }
                
                # AddPromptTab
                tab_names = [f"📦 {name}" for name in roles.values()] + ["💬 Prompt"]
                tabs = st.tabs(tab_names)
                
                # RoleDataTab
                for i, (role_key, role_name) in enumerate(roles.items()):
                    with tabs[i]:
                        st.markdown(f"### {role_name}OperationsData")
                        
                        # BaseOperationsData
                        role_data = pd.DataFrame([
                            {
                                "Round": round_data['round'],
                                "Received Demand": round_data['agents'][role_key].get('demand_received', 0),
                                "DownSingleDecision": round_data['agents'][role_key].get('order_placed', 0),
                                "Period StartInventory": self.format_inventory_display(round_data['agents'][role_key]['start_state'].get('inventory', 0)),
                                "Period EndInventory": self.format_inventory_display(round_data['agents'][role_key]['end_state'].get('inventory', 0)),
                                ("Production In Progress Total" if role_key == 'manufacturer' else "In-Transit Inventory"): round_data['agents'][role_key]['end_state'].get('total_in_transit', 0),
                                ("Next Period Completed Production Quantity" if role_key == 'manufacturer' else "Soon to Arrive"): round_data['agents'][role_key]['end_state'].get('incoming_shipment', 0),
                                "StockoutQuantity": self.format_backorder_display(round_data['agents'][role_key]['end_state'].get('backorder', 0)),
                                "Holding Cost": round_data['agents'][role_key]['end_state'].get('holding_cost', 0),
                                "Stockout Cost": round_data['agents'][role_key]['end_state'].get('shortage_cost', 0),
                                "Total Cost": round_data['agents'][role_key].get('round_cost', 0)
                            }
                            for round_data in st.session_state.result.round_history
                            if role_key in round_data['agents']
                        ])
                        st.dataframe(role_data, width='stretch')
                        
                        # In-Transit Inventory Details
                        st.markdown(f"#### {role_name}{'Production In Progress Details' if role_key == 'manufacturer' else 'In-Transit Inventory Details'}")
                        pipeline_data = []
                        for round_data in st.session_state.result.round_history:
                            if role_key in round_data['agents']:
                                pipeline = round_data['agents'][role_key]['end_state'].get('shipment_pipeline', [])
                                if role_key == 'manufacturer':
                                    # Complete display of manufacturer production pipeline: zero-padded to Production Lead Time length, shown per period (including 0)
                                    lt = getattr(st.session_state.result.config.simulation, 'manufacturer_lead_time', None)
                                    lt = lt if isinstance(lt, int) and lt > 0 else (len(pipeline) if pipeline else 0)
                                    full_len = max(lt, len(pipeline))
                                    full_pipeline = [pipeline[i] if i < len(pipeline) else 0 for i in range(full_len)]
                                    pipeline_str = ', '.join([f"No.{i+1}Period EstimatedCompleted:{qty}" for i, qty in enumerate(full_pipeline)])
                                    total_pipeline_sum = sum(full_pipeline)
                                else:
                                    pipeline_str = ', '.join([f"No.{i+1}Period EstimatedtoArrival:{qty}" for i, qty in enumerate(pipeline) if qty > 0])
                                    total_pipeline_sum = sum(pipeline)
                                if not pipeline_str:
                                    pipeline_str = "NoneinProductionProduction" if role_key == 'manufacturer' else "NoneIn-Transit Inventory"
                                pipeline_data.append({
                                    "Round": round_data['round'],
                                    ("Production In Progress Details" if role_key == 'manufacturer' else "In-Transit Inventory Details"): pipeline_str,
                                    ("Total In Production Quantity" if role_key == 'manufacturer' else "Total In Transit Quantity"): total_pipeline_sum
                                })
                        
                        if pipeline_data:
                            pipeline_df = pd.DataFrame(pipeline_data)
                            st.dataframe(pipeline_df, width='stretch')
                        
                        # OrderFlowInfo
                        st.markdown(f"#### {role_name}OrderFlowInfo")
                        order_flow_data = []
                        for round_data in st.session_state.result.round_history:
                            # FindSent toCurrentRoleOrder
                            received_orders = []
                            for order_flow in round_data.get('orders_flow', []):
                                if order_flow.get('to') == role_key:
                                    received_orders.append(f"From {order_flow.get('from', 'Unknown')}:{order_flow.get('quantity', 0)}")
                            
                            # FindCurrentRoleIssued Order
                            sent_orders = []
                            for order_flow in round_data.get('orders_flow', []):
                                if order_flow.get('from') == role_key:
                                    sent_orders.append(f"Sent to {order_flow.get('to', 'Unknown')}: {order_flow.get('quantity', 0)}")
                            
                            order_flow_data.append({
                                "Round": round_data['round'],
                                "Received Order": ', '.join(received_orders) if received_orders else "None",
                                "Issued Order": ', '.join(sent_orders) if sent_orders else "None"
                            })
                        
                        if order_flow_data:
                            order_flow_df = pd.DataFrame(order_flow_data)
                            st.dataframe(order_flow_df, width='stretch')
                        
                        # DataDownloadFeature
                        csv = role_data.to_csv(index=False, encoding='utf-8-sig')
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label=f"📥 Download{role_name}Data",
                            data=csv,
                            file_name=f"{role_name}_OperationsData_{timestamp}.csv",
                            mime="text/csv"
                        )
                
                # PromptTab
                with tabs[len(roles)]:
                    st.markdown("### 💬 Per Round Prompt Details")
                    st.markdown("View each participant's complete prompt content for each round")
                    
                    # Round Selector
                    available_rounds = [rd['round'] for rd in st.session_state.result.round_history if 'prompts' in rd]
                    if available_rounds:
                        selected_round = st.selectbox(
                            "Select Round",
                            available_rounds,
                            format_func=lambda x: f"Round{x} Round"
                        )
                        
                        # FindtoSelectinRoundData
                        selected_round_data = None
                        for rd in st.session_state.result.round_history:
                            if rd['round'] == selected_round and 'prompts' in rd:
                                selected_round_data = rd
                                break
                        
                        if selected_round_data and 'prompts' in selected_round_data:
                            st.markdown(f"#### Round{selected_round} RoundPrompt")
                            
                            # RoleSelectSelector
                            available_roles = list(selected_round_data['prompts'].keys())
                            role_names_map = {
                                'retailer': 'Retailer',
                                'wholesaler': 'Wholesaler', 
                                'distributor': 'Distributor',
                                'manufacturer': 'Manufacturer'
                            }
                            
                            selected_role = st.selectbox(
                                "Select Participants",
                                available_roles,
                                format_func=lambda x: role_names_map.get(x, x)
                            )
                            
                            if selected_role in selected_round_data['prompts']:
                                prompt_data = selected_round_data['prompts'][selected_role]
                                
                                # DisplayPromptContent
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("##### 🤖 SystemPrompt")
                                    st.text_area(
                                        "SystemPromptContent",
                                        value=prompt_data.get('system_prompt', ''),
                                        height=300,
                                        key=f"system_prompt_{selected_round}_{selected_role}",
                                        label_visibility="collapsed"
                                    )
                                
                                with col2:
                                    st.markdown("##### 👤 UserPrompt")
                                    st.text_area(
                                        "UserPromptContent",
                                        value=prompt_data.get('user_prompt', ''),
                                        height=300,
                                        key=f"user_prompt_{selected_round}_{selected_role}",
                                        label_visibility="collapsed"
                                    )
                                
                                # CompletePromptDisplay
                                with st.expander("ViewCompletePrompt", expanded=False):
                                    st.text_area(
                                        "CompletePrompt",
                                        value=prompt_data.get('full_prompt', ''),
                                        height=400,
                                        key=f"full_prompt_{selected_round}_{selected_role}",
                                        label_visibility="collapsed"
                                    )
                                
                                # DisplayDecision Explanation
                                st.markdown("##### 🎯 Decision Explanation")
                                decision_explanation = ""
                                if selected_role in selected_round_data['agents']:
                                    decision_explanation = selected_round_data['agents'][selected_role].get('decision_explanation', 'No decision explanation provided')
                                
                                if decision_explanation:
                                    st.info(f"**Decision Explanation:** {decision_explanation}")
                                else:
                                    st.warning("No decision explanation info received for this round")
                                
                                # Download Prompt and Decision Explanation
                                prompt_text = f"""Round{selected_round} Round - {role_names_map.get(selected_role, selected_role)} PromptandDecision Explanation

SystemPrompt:
{prompt_data.get('system_prompt', '')}

UserPrompt:
{prompt_data.get('user_prompt', '')}

CompletePrompt:
{prompt_data.get('full_prompt', '')}

Decision Explanation:
{decision_explanation}"""
                                
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                st.download_button(
                                    label=f"📥 Download Prompt and Decision Explanation",
                                    data=prompt_text,
                                    file_name=f"PromptDecision Explanation_No.{selected_round}Round_{role_names_map.get(selected_role, selected_role)}_{timestamp}.txt",
                                    mime="text/plain"
                                )
                        else:
                            st.warning(f"Round{selected_round} RoundNoPromptData")
                    else:
                        st.info("Current simulation results have no prompt data. Please re-run the simulation to collect prompt info.")
    
    def show_example_results_loader(self):
        """Display example result load interface"""
        st.title("📊 Load History Simulation Results")
        st.markdown("---")
        
        # ReturnButton
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Return to Main Interface", type="secondary"):
                st.session_state.show_example = False
                st.rerun()
        
        # ScanHistoryFile
        history_files = self.scan_history_files()
        
        if not history_files:
            st.warning("📁 No history simulation result files found")
            st.markdown("""
            **PossibleReason:**
            - Have notNoRunyetSimulation
            - Simulation ResultsFilewas movedorDelete
            - FileSaveinOtherDirectory
            
            **Suggestions:**
            1. FirstRunAtimesSimulationGenerateResultFile
            2. CheckProject rootDirectoryDown `simulation_outputs` Filefolder
            """)
            return
        
        st.success(f"🎉 Findto {len(history_files)} History Simulation ResultsFile")
        
        # FileSelect
        st.markdown("### 📋 Select to loadSimulation Results")
        
        # CreateFileSelectOption
        file_options = []
        for file_info in history_files:
            display_name = f"{file_info['display_name']} ({file_info['file_size']})"
            file_options.append(display_name)
        
        selected_index = st.selectbox(
            "SelectFile:",
            range(len(file_options)),
            format_func=lambda x: file_options[x],
            help="Select to loadHistory Simulation ResultsFile"
        )
        
        if selected_index is not None:
            selected_file = history_files[selected_index]
            
            # DisplayFileDetails
            st.markdown("### 📄 FileDetails")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("FileName", selected_file['filename'])
            with col2:
                st.metric("CreateWhenTime", selected_file['created_time'])
            with col3:
                st.metric("FileSize", selected_file['file_size'])
            
            # LoadButton
            if st.button("🚀 Load this Simulation Result", type="primary", width='stretch'):
                with st.spinner("PositiveinLoadSimulation Results..."):
                    success = self.load_simulation_result(selected_file['filepath'])
                    
                    if success:
                        st.success("✅ Simulation result loaded successfully！")
                        st.session_state.show_example = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Simulation result load failed，PleaseCheckFileformat")
    
    def scan_history_files(self):
        """Scan history simulation results files"""
        import os
        import json
        from datetime import datetime
        from pathlib import Path
        
        # ScanMultiPossibleDirectory
        search_dirs = [
            "simulation_outputs",
            ".",  # CurrentDirectory
            "demo_comparison",
            "outputs"
        ]
        
        history_files = []
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for filename in os.listdir(search_dir):
                    if filename.endswith('.json') and ('simulation_result' in filename or 'result' in filename):
                        filepath = os.path.join(search_dir, filename)
                        
                        try:
                            # GetFileInfo
                            file_stat = os.stat(filepath)
                            file_size = f"{file_stat.st_size / 1024:.1f} KB"
                            created_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            
                            # TryReadFileContentGetMoreInfo
                            display_name = filename
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    
                                # TryfromFileContentinExtractfriendlierDisplayName
                                if 'metadata' in data and 'scenario_name' in data['metadata']:
                                    scenario_name = data['metadata']['scenario_name']
                                    display_name = f"{scenario_name} - {filename}"
                                elif 'config' in data and 'demand_pattern' in data['config']:
                                    pattern = data['config']['demand_pattern']
                                    display_name = f"{pattern}Demand - {filename}"
                                    
                            except:
                                pass  # IfNoneMethodParse，Use original filename
                            
                            history_files.append({
                                'filename': filename,
                                'filepath': filepath,
                                'display_name': display_name,
                                'created_time': created_time,
                                'file_size': file_size,
                                'modified_time': file_stat.st_mtime
                            })
                            
                        except Exception as e:
                            continue  # Skip unreadable files
        
        # Sorted by modification time descending (most recent first)
        history_files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return history_files
    
    def load_simulation_result(self, filepath):
        """Load simulation results file"""
        try:
            import json
            # Remove relative import, directly use simulated object
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Based onFileformatReconstructSimulationResultObject
            if 'round_history' in data:
                # StandardformatSimulation ResultsFile
                result = self.reconstruct_simulation_result(data)
                bullwhip_metrics = data.get('bullwhip_metrics', {})
                
            elif 'results' in data and 'bullwhip_metrics' in data:
                # demo formatSimulation ResultsFile
                result = self.reconstruct_from_demo_format(data)
                bullwhip_metrics = data['bullwhip_metrics']
                
            elif 'run' in data and 'timestamp' in data and len(data) <= 3:
                # SimplifiedformatFile，OnlyhasbasicThisInfo
                st.warning("⚠️ This is a simplified simulation record file，Does not contain complete simulation data")
                st.info("""**FileInfo:**
                - RunRun #:{}
                - WhenTimestamp:{}
                
                **Suggestions:**
                - SelectContains complete simulation dataFile
                - FindFileNameContainsMoreInfoResultFile""".format(data.get('run', 'N/A'), data.get('timestamp', 'N/A')))
                return False
                
            else:
                st.error("❌ Unsupported file format")
                st.info("""**SupportFileformat:**
                - Contains `round_history` Standard simulation results file
                - Contains `results` and `bullwhip_metrics` demo formatFile
                
                **CurrentFileContainsField:**
                {}""".format(', '.join(data.keys()) if isinstance(data, dict) else 'NoneMethodParse'))
                return False
            
            # Savetosession state
            st.session_state.result = result
            st.session_state.bullwhip_metrics = bullwhip_metrics
            
            return True
            
        except Exception as e:
            st.error(f"Error when loading file: {str(e)}")
            return False
    
    def reconstruct_simulation_result(self, data):
        """Reconstruct simulation result object from standard format"""
        # Create simulated SimulationResult object, not dependent on original class
        class MockSimulationResult:
            def __init__(self, data):
                self.round_history = data.get('round_history', [])
                self.total_cost = data.get('total_cost', 0)
                self.agent_costs = data.get('agent_costs', {})
                self.agent_states = data.get('agent_states', {})
                self.bullwhip_metrics = data.get('bullwhip_metrics', {})
                self.simulation_time = data.get('simulation_time', 0)
                self.config = data.get('config', {})
                # IfNoagent_states，fromround_historyinReconstruct
                if not self.agent_states and self.round_history:
                    last_round = self.round_history[-1] if self.round_history else {}
                    self.agent_states = {
                        'retailer': last_round.get('retailer', {}),
                        'wholesaler': last_round.get('wholesaler', {}),
                        'distributor': last_round.get('distributor', {}),
                        'manufacturer': last_round.get('manufacturer', {})
                    }
                
                self.total_rounds = len(self.round_history)
        
        return MockSimulationResult(data)
    
    def reconstruct_from_demo_format(self, data):
        """Reconstruct simulation result object from demo format"""
        # CreateSimulateSimulation Results
        class MockSimulationResult:
            def __init__(self, data):
                results = data['results']
                metadata = data.get('metadata', {})
                
                self.total_cost = results.get('total_cost', 0)
                self.total_rounds = results.get('rounds', 20)
                self.agent_costs = results.get('agents', {})
                self.simulation_time = 0
                # CreateSimulateround_history
                self.round_history = []
                for i in range(self.total_rounds):
                    round_data = {
                        'round': i + 1,
                        'retailer': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('retailer', {}).get('cost', 0) / self.total_rounds},
                        'wholesaler': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('wholesaler', {}).get('cost', 0) / self.total_rounds},
                        'distributor': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('distributor', {}).get('cost', 0) / self.total_rounds},
                        'manufacturer': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('manufacturer', {}).get('cost', 0) / self.total_rounds},
                        'market_demand': 10 + (i % 5)  # SimulateDemandChange
                    }
                    self.round_history.append(round_data)
                
                # Createagent_states
                self.agent_states = {
                    'retailer': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('retailer', {}).get('cost', 0)},
                    'wholesaler': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('wholesaler', {}).get('cost', 0)},
                    'distributor': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('distributor', {}).get('cost', 0)},
                    'manufacturer': {'inventory': 10, 'backorder': 0, 'cost': self.agent_costs.get('manufacturer', {}).get('cost', 0)}
                }
                
                # CreateSimulateconfig
                self.config = {
                    'scenario_name': metadata.get('scenario_name', 'HistorySimulation'),
                    'strategy': metadata.get('strategy', 'unknown'),
                    'demand_pattern': 'HistoryData',
                    'total_weeks': self.total_rounds
                }
        
        return MockSimulationResult(data)
    
    def display_prompt_debug_panel(self, result, ui_config):
        """Display prompt debug panel"""
        st.markdown("### 🔧 Prompt Debug Panel")
        st.markdown("This panel allows you to view and edit prompts used by each role in each round, to adjust and optimize decision logic.")
        
        if not hasattr(result, 'round_history') or not result.round_history:
            st.info("ℹ️ PleaseFirstRun SimulationtoGeneratePromptData")
            st.markdown("""
            **UseDescription:**
            1. Configure Simulation Parameters in the left sidebar
            2. EnsureCheckDone"Enable Prompt Debugging"
            3. PointClick"🚀 StartSimulation"Run Simulation
            4. After simulation completes, view prompt details of each round here
            
            **FeatureFeaness:**
            - 📊 **Round Selector**:ViewanyCompletedRoundPrompt
            - 👥 **RoleSelectSelector**:ViewFourSupply Chain RolePrompt
            - 📝 **PromptDisplay**:ViewSystemPromptandUserPrompt
            - 🤖 **LLM Response**:ViewModelCompleteResponse
            - 💭 **Decision Explanation**:ViewDecision ReasoningProcess
            - ✏️ **Edit Feature**:Allow Prompt Editing（needEnableEditMode）
            - 📤 **ExportFeature**:ExportPromptDataForAnalysis
            """)
            return
        
        # Round Selector
        col1, col2 = st.columns(2)
        with col1:
            selected_round = st.selectbox(
                "Select Round",
                range(1, len(result.round_history) + 1),
                format_func=lambda x: f"Round{x} Round",
                key="debug_round_selector"
            )
        
        with col2:
            selected_role = st.selectbox(
                "Select Role",
                ["retailer", "wholesaler", "distributor", "manufacturer"],
                format_func=lambda x: {"retailer": "Retailer", "wholesaler": "Wholesaler", 
                                     "distributor": "Distributor", "manufacturer": "Manufacturer"}[x],
                key="debug_role_selector"
            )
        
        # GetSelectinRoundData
        round_data = result.round_history[selected_round - 1]
        role_name_map = {"retailer": "Retailer", "wholesaler": "Wholesaler", 
                        "distributor": "Distributor", "manufacturer": "Manufacturer"}
        
        st.markdown(f"#### {role_name_map[selected_role]} - No.{selected_round}Round")
        
        # CheckWhether there isPromptData
        if selected_role not in round_data.get('agents', {}):
            st.warning(f"⚠️ No.{selected_round}RoundNo{role_name_map[selected_role]}Data")
            return
        
        agent_data = round_data['agents'][selected_role]
        
        # DisplayDecisionInfo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Order Decision", agent_data.get('order_placed', 'N/A'))
        with col2:
            st.metric("Current Inventory", agent_data.get('end_state', {}).get('inventory', 'N/A'))
        with col3:
            st.metric("Received Demand", agent_data.get('demand_received', 'N/A'))
        
        # PromptDisplayandEdit
        st.markdown("##### 📝 PromptContent")
        
        # CheckWhether there isPromptData
        prompt_data = {}
        if 'prompts' in round_data and selected_role in round_data['prompts']:
            prompt_data = round_data['prompts'][selected_role]
        elif 'prompt_info' in agent_data:
            prompt_data = agent_data['prompt_info']
        
        if not prompt_data:
            st.info("ℹ️ No prompt info recorded for this round（PossibleisRule-BasedorDataNotSave）")
            return
        
        # SystemPrompt
        with st.expander("🤖 SystemPrompt", expanded=True):
            system_prompt = prompt_data.get('system_prompt', 'No system prompt recorded')
            if ui_config.get('debug_allow_edit', False):
                edited_system = st.text_area(
                    "Edit System Prompt",
                    value=system_prompt,
                    height=200,
                    key=f"edit_system_prompt_{selected_round}_{selected_role}"
                )
                if st.button(f"💾 Save System Prompt Changes", key=f"save_system_{selected_round}_{selected_role}"):
                    st.success("✅ System prompt changes saved（Note:ThisjustInterfaceDisplay，Notwill affectCompletedSimulation）")
            else:
                st.code(system_prompt, language="text")
        
        # UserPrompt
        with st.expander("👤 UserPrompt", expanded=True):
            user_prompt = prompt_data.get('user_prompt', 'No user prompt recorded')
            if ui_config.get('debug_allow_edit', False):
                edited_user = st.text_area(
                    "Edit User Prompt",
                    value=user_prompt,
                    height=300,
                    key=f"edit_user_prompt_{selected_round}_{selected_role}"
                )
                if st.button(f"💾 Save User Prompt Changes", key=f"save_user_{selected_round}_{selected_role}"):
                    st.success("✅ User prompt changes saved（Note:ThisjustInterfaceDisplay，Notwill affectCompletedSimulation）")
            else:
                st.code(user_prompt, language="text")
        
        # LLM Response
        with st.expander("🤖 LLM Response", expanded=False):
            llm_response = prompt_data.get('llm_response', agent_data.get('llm_response', 'No LLM response recorded'))
            st.code(llm_response, language="text")
        
        # Decision Explanation
        with st.expander("💭 Decision Explanation", expanded=False):
            decision_explanation = agent_data.get('decision_explanation', prompt_data.get('decision_explanation', 'No decision explanation recorded'))
            st.markdown(decision_explanation)
        
        # Real-timeDebug Features
        if ui_config.get('debug_realtime', False):
            st.markdown("---")
            st.markdown("##### ⚡ Real-time Debug Features")
            st.info("💡 Real-time debug features already enabled. Prompt info will be displayed during simulation.")
            
            # Here you can add real-time adjustment related features
            if st.button("🔄 Refresh Current Data", key=f"refresh_{selected_round}_{selected_role}"):
                st.rerun()
        
        # ExportFeature
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Current Prompt", key=f"export_{selected_round}_{selected_role}"):
                export_data = {
                    "round": selected_round,
                    "role": selected_role,
                    "role_name": role_name_map[selected_role],
                    "prompt_data": prompt_data,
                    "decision": agent_data.get('order_placed', 'N/A')
                }
                
                import json
                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 DownloadJSONFile",
                    data=json_str,
                    file_name=f"prompt_debug_{role_name_map[selected_role]}_round{selected_round}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export All Round Prompts", key=f"export_all_{selected_role}"):
                all_prompts = []
                for i, round_data in enumerate(result.round_history):
                    if selected_role in round_data.get('agents', {}):
                        agent_data = round_data['agents'][selected_role]
                        prompt_data = agent_data.get('prompt_info', {})
                        if prompt_data:
                            all_prompts.append({
                                "round": i + 1,
                                "role": selected_role,
                                "role_name": role_name_map[selected_role],
                                "prompt_data": prompt_data,
                                "decision": agent_data.get('order_placed', 'N/A')
                            })
                
                if all_prompts:
                    import json
                    json_str = json.dumps(all_prompts, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="💾 DownloadCompleteJSONFile",
                        data=json_str,
                        file_name=f"all_prompts_{role_name_map[selected_role]}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("⚠️ No prompt data found")
    
    def _validate_unstable_demand_params(self, demand_type: str, preview_demands: list) -> list:
        """Validate volatile demand pattern parameters"""
        validation_results = []
        
        if demand_type == "Autoregressive":
            # CheckAutocorrelation
            if len(preview_demands) > 10:
                autocorr = np.corrcoef(preview_demands[:-1], preview_demands[1:])[0, 1]
                if not np.isnan(autocorr):
                    if autocorr > 0.7:
                        validation_results.append(("⚠️", "Autocorrelation", f"Autocorrelation Coefficient {autocorr:.3f} Too High，May lead toOver-smoothing"))
                    elif autocorr < 0.1:
                        validation_results.append(("ℹ️", "Autocorrelation", f"Autocorrelation Coefficient {autocorr:.3f} Low，Time dependency is not obvious"))
                    else:
                        validation_results.append(("✅", "Autocorrelation", f"Autocorrelation Coefficient {autocorr:.3f} Moderate"))
        
        elif demand_type == "ARMA Demand":
            # Check ARMA model stability
            if len(preview_demands) > 15:
                # Check if variance is stable
                first_half_var = np.var(preview_demands[:len(preview_demands)//2])
                second_half_var = np.var(preview_demands[len(preview_demands)//2:])
                var_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
                
                if var_ratio > 3.0:
                    validation_results.append(("⚠️", "Variance Stability", f"Before/After Halves Variance Ratio {var_ratio:.2f} Too large，Model is volatile"))
                else:
                    validation_results.append(("✅", "Variance Stability", f"Before/After Halves Variance Ratio {var_ratio:.2f} Stable"))
        
        elif demand_type == "Jump Diffusion":
            # CheckJump Frequencyandmagnitude
            if len(preview_demands) > 5:
                diffs = np.diff(preview_demands)
                large_jumps = np.abs(diffs) > 2 * np.std(diffs)
                jump_count = np.sum(large_jumps)
                jump_rate = jump_count / len(diffs)
                
                if jump_rate > 0.3:
                    validation_results.append(("⚠️", "Jump Frequency", f"Jump Frequency {jump_rate:.2%} Too High，Demand is too volatile"))
                elif jump_rate < 0.05:
                    validation_results.append(("ℹ️", "Jump Frequency", f"Jump Frequency {jump_rate:.2%} Low，Jump effect is not obvious"))
                else:
                    validation_results.append(("✅", "Jump Frequency", f"Jump Frequency {jump_rate:.2%} Moderate"))
        
        elif demand_type == "Poisson Jump":
            # Check jump discreteness
            if len(preview_demands) > 10:
                unique_values = len(np.unique(preview_demands))
                total_values = len(preview_demands)
                diversity_ratio = unique_values / total_values
                
                if diversity_ratio < 0.3:
                    validation_results.append(("ℹ️", "Value Diversity", f"Unique Value Ratio {diversity_ratio:.2%}，Demand values are relatively concentrated"))
                else:
                    validation_results.append(("✅", "Value Diversity", f"Unique Value Ratio {diversity_ratio:.2%}，Demand value distribution is reasonable"))
        
        elif demand_type == "Regime Switching":
            # CheckRegime ConversionObviousness
            if len(preview_demands) > 20:
                # Simple single regime detection: find significant mean change
                mid_point = len(preview_demands) // 2
                first_half_mean = np.mean(preview_demands[:mid_point])
                second_half_mean = np.mean(preview_demands[mid_point:])
                mean_diff = abs(second_half_mean - first_half_mean)
                
                if mean_diff > 3:
                    validation_results.append(("✅", "Regime Conversion", f"DetecttoObviousRegime Conversion，MeanDifference {mean_diff:.2f}"))
                else:
                    validation_results.append(("ℹ️", "Regime Conversion", f"Regime ConversionNotObvious，MeanDifference {mean_diff:.2f}"))
        
        elif demand_type == "Volatility Clustering":
            # Check volatility clustering effect
            if len(preview_demands) > 15:
                # Calculate rolling standard deviation
                window_size = 5
                rolling_stds = []
                for i in range(window_size, len(preview_demands)):
                    rolling_std = np.std(preview_demands[i-window_size:i])
                    rolling_stds.append(rolling_std)
                
                if len(rolling_stds) > 5:
                    # CheckVolatility Rate Fluctuation
                    volatility_of_volatility = np.std(rolling_stds)
                    if volatility_of_volatility > 1.0:
                        validation_results.append(("✅", "Volatility Rate Clustering", f"Detected volatility clustering effect，Volatility Rate Fluctuation {volatility_of_volatility:.2f}"))
                    else:
                        validation_results.append(("ℹ️", "Volatility Rate Clustering", f"Volatility clustering effect not evident，Volatility Rate Fluctuation {volatility_of_volatility:.2f}"))
        
        return validation_results
    
    def _show_unstable_demand_suggestions(self, demand_type: str, avg_demand: float, cv: float):
        """Display volatile demand pattern special suggestions"""
        if demand_type == "Autoregressive":
            st.info("💡 Autoregressive Suggestions: Consider using a relatively long history data window for forecasting. Enable Information Sharing to improve Forecast Accuracy.")
        
        elif demand_type == "ARMA Demand":
            st.info("💡 ARMA DemandSuggestions:Thistype ofDemand PatternCombinedDoneHistoryTrendandRandom shocks，We recommend usingadaptive order strategy")
        
        elif demand_type == "Jump Diffusion":
            st.warning("⚠️ Jump Diffusion Warning: Demand may appear with sudden jumps. Suggest increasing safety inventory and enabling rapid response mechanisms.")
        
        elif demand_type == "Poisson Jump":
            st.info("💡 Poisson JumpSuggestions:DemandJumpFollows PoissonProcess，We recommend usingBased onProbabilityInventoryManagementStrategy")
        
        elif demand_type == "Regime Switching":
            st.warning("⚠️ Regime SwitchingWarning:Demand may switch between different states，We recommend usingMultiStatusForecastModelandflexibleInventoryStrategy")
        
        elif demand_type == "Volatility Clustering":
            st.info("💡 Volatility ClusteringSuggestions:DemandFluctuationnesswillClusteringappears，We suggest increasing inventory buffer in high-fluctuation periods and optimizing cost in low-fluctuation periods")
        
        # GeneralSuggestions
        if cv > 1.0:
            st.warning("⚠️ Unstable Demand General Suggestions: CV is very high. Strongly suggest enabling Information Sharing to improve Supply Chain Stability.")

    def _show_unstable_demand_analysis(self, demand_pattern: str, preview_demands: list):
        """Display volatile demand pattern characteristics analysis"""
        import numpy as np
        from scipy import stats
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # AutocorrelationAnalysis
            if len(preview_demands) > 10:
                autocorr = np.corrcoef(preview_demands[:-1], preview_demands[1:])[0, 1]
                if not np.isnan(autocorr):
                    st.metric("Autocorrelation Coefficient", f"{autocorr:.3f}")
                else:
                    st.metric("Autocorrelation Coefficient", "N/A")
            
            # CV
            cv = np.std(preview_demands) / np.mean(preview_demands) if np.mean(preview_demands) > 0 else 0
            st.metric("CV", f"{cv:.3f}")
        
        with col2:
            # SkewnessandKurtosis
            skewness = stats.skew(preview_demands)
            kurtosis = stats.kurtosis(preview_demands)
            st.metric("Skewness", f"{skewness:.3f}")
            st.metric("Kurtosis", f"{kurtosis:.3f}")
        
        with col3:
            # JumpDetect（SimpleSingleVerThis）
            if len(preview_demands) > 5:
                diffs = np.diff(preview_demands)
                large_jumps = np.sum(np.abs(diffs) > 2 * np.std(diffs))
                st.metric("Large Jump Count", f"{large_jumps}")
            
            # TrendDetect
            if len(preview_demands) > 10:
                slope, _, r_value, p_value, _ = stats.linregress(range(len(preview_demands)), preview_demands)
                if p_value < 0.05:
                    trend_strength = "Strong" if abs(r_value) > 0.7 else "in" if abs(r_value) > 0.3 else "Weak"
                    trend_direction = "Rising" if slope > 0 else "Falling"
                    st.metric("Trend", f"{trend_direction}({trend_strength})")
                else:
                    st.metric("Trend", "No Significant Trend")
        
        # ModeSpecificAnalysis
        if demand_pattern == "Autoregressive":
            st.info("📈 AutoregressiveCharacteristics:Correlation between current and historical demand")
        elif demand_pattern == "ARMA Demand":
            st.info("📊 ARMACharacteristics:CombinedHistoryTrendandRandom shock compositeMode")
        elif demand_pattern == "Jump Diffusion":
            st.info("🚀 JumpDiffusionCharacteristics:Sudden jumps within continuous change")
        elif demand_pattern == "Poisson Jump":
            st.info("⚡ Poisson JumpCharacteristics:Demand jumps at random time intervals")
        elif demand_pattern == "Regime Switching":
            st.info("🔄 Regime ConversionCharacteristics:Switching of demand between different states")
        elif demand_pattern == "Volatility Clustering":
            st.info("📈 Volatility Rate ClusteringCharacteristics:Temporal clustering effect of volatility")


def main():
    """Main function entry point"""
    app = StreamlitBeerGameApp()
    app.run_app()


if __name__ == "__main__":
    main()