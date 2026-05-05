# LLM Beer Game — Multi-Agent Supply Chain Simulation with Order Constraints

A Python-based Beer Distribution Game simulation where **LLM-powered agents** (Retailer → Wholesaler → Distributor → Manufacturer) make autonomous ordering decisions under a **multi-layered order constraint system** — the core innovation for mitigating the bullwhip effect. An **AI Coordinator** provides global analysis and per-role guidance each round.

## Table of Contents

- [Core Innovation: Order Constraint System](#core-innovation-order-constraint-system)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [LLM Provider Setup](#llm-provider-setup)
- [Configuration](#configuration)
- [Demand Patterns](#demand-patterns)
- [CLI Usage](#cli-usage)
- [Supported LLM Providers](#supported-llm-providers)
- [Project Structure](#project-structure)
- [Analysis & Visualization](#analysis--visualization)
- [References](#references)

## Core Innovation: Order Constraint System

The bullwhip effect — where small fluctuations in customer demand amplify into increasingly large order swings upstream — is the central problem in supply chain management. This project tackles it through a **three-layer order constraint architecture** that bounds ordering behavior at every tier.

### Three-Layer Constraint Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Adaptive Order Limits (Dynamic)               │
│  Automatically adjusts [min, max] per round based on    │
│  actual retailer demand using ratio/window/forecast     │
│  algorithms. Upstream roles get progressively wider     │
│  bands via role multipliers (×1.0 → ×1.3).              │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Discrete Order Points (Semi-Fixed)            │
│  Restricts orders to a predefined set of allowed        │
│  values (e.g., {5, 10, 15, 20}). Snaps LLM output to   │
│  the nearest valid point. Takes priority over Layer 3.  │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Fixed Min/Max Bounds (Static)                 │
│  Per-role hard floor and ceiling on order quantities.   │
│  Serves as a safety net and validation boundary when    │
│  higher layers are active.                              │
└─────────────────────────────────────────────────────────┘
         +
┌─────────────────────────────────────────────────────────┐
│  Soft Layer: Coordinator AI Guidance                    │
│  The coordinator analyzes global supply chain state     │
│  and suggests specific order quantities to each role.   │
│  Injected into LLM prompts as natural language guidance. │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Fixed Min/Max Order Limits

Each of the 4 roles can have independent `[min_order, max_order]` bounds. The LLM agent is instructed about these limits in both its system prompt and user prompt. Any order outside the range is automatically clamped:

```
Order limits: minimum order quantity 3, maximum order quantity 8
```

When no limits are configured, the effective range is `[0, 1000]` — essentially unconstrained. Limits can be toggled on/off independently per simulation run.

### Layer 2: Discrete Order Point Selection

Instead of a continuous range, orders can be restricted to a fixed set of allowed values. For example, `{5, 10, 15, 20}` forces the agent to choose only from these quantities. The system snaps any LLM output to the nearest valid point:

- The LLM is prompted: "Your order quantity MUST be selected from the following list: [5, 10, 15, 20]"
- Post-processing enforces: `argmin(|order - point|)` for each point in the set
- When both Layer 1 and Layer 2 are active, discrete points take priority; range limits serve as validation boundaries

### Layer 3: Adaptive Order Limits (Key Innovation)

Adaptive limits **dynamically recompute** the allowed `[min, max]` range each round based on actual end-customer (retailer) demand. This prevents agents from overreacting to temporary demand fluctuations — the core mechanism of bullwhip mitigation.

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `false` | Master switch for adaptive limits |
| `base_min_order` | 3 | Floor of the adaptive range |
| `base_max_order` | 8 | Ceiling of the adaptive range |
| `method` | `ratio` | Algorithm: `ratio`, `window`, or `forecast` |
| `adaptation_rate` | 0.3 | Smoothing factor (0–1) for limit transitions |
| `min_absolute_limit` | 1 | Hard floor — no limit can go below this |
| `max_absolute_limit` | 100 | Hard ceiling — no limit can go above this |

**Upstream amplification coefficients:** As orders move upstream, slightly wider bands are allowed to accommodate genuine supply-demand mismatches:

| Role | Multiplier | Rationale |
|---|---|---|
| Retailer | ×1.0 | Closest to real demand — tightest bounds |
| Wholesaler | ×1.1 | Moderate buffer for retailer aggregation |
| Distributor | ×1.2 | Wider band for multi-tier variability |
| Manufacturer | ×1.3 | Widest band to absorb production lead time |

**Three adaptive algorithms:**

1. **Ratio-Based** (`method='ratio'`): Computes `demand_ratio = current_demand / average_demand`, then scales base limits by `ratio × min_ratio × role_factor` and `ratio × max_ratio × role_factor`. Best for stable demand patterns.

2. **Window-Based** (`method='window'`): Slides a window over recent demand history (`window_size=5`), computes mean and standard deviation, then sets `min = mean - α×σ` and `max = mean + β×σ`, multiplied by the role factor. Best for volatile demand.

3. **Forecast-Based** (`method='forecast'`): Uses a simple moving average of the last 3 periods plus a trend component, then applies fixed multipliers (0.7 for min, 1.5 for max). Best for trending demand.

All methods apply **exponential smoothing** between rounds via `adaptation_rate` to prevent abrupt limit changes that could destabilize the chain.

### How This Mitigates the Bullwhip Effect

```
Without Constraints:          With Adaptive Constraints:
Demand: 5  5  5  8  5         Demand: 5  5  5  8  5
Retail:  5  5  5  12 3        Retail:  5  5  5  7  5    ← clamped
Whole:   5  5  8  20 2        Whole:   5  5  6  8  5    ← smoothed
Dist:    5  8  15 30 1        Dist:    5  6  7  9  5    ← smoothed
Manuf:   8  15 30 50 0        Manuf:   6  7  8  10 5    ← smoothed
Bullwhip: SEVERE              Bullwhip: CONTROLLED
```

The adaptive system recognizes that a single-period demand spike (5→8) should NOT trigger a 12-unit order from the retailer, which cascades into a 50-unit production run at the manufacturer. Instead, limits tighten around the observed demand range, and upstream roles receive progressively relaxed but still-bounded ranges.

### Coordinator Soft Guidance

As a complementary soft layer, the **AI Coordinator** analyzes the global supply chain state each round and outputs per-role order quantity suggestions with justifications. This guidance is injected into each agent's LLM prompt as natural language, allowing the agent to consider system-wide conditions when making its decision — without hard-coding a specific coordination policy.

## Key Features

- **Three-Layer Order Constraint System** — Fixed min/max bounds, discrete point selection, and adaptive dynamic limits with 3 algorithms (ratio, window, forecast) — the primary mechanism for bullwhip effect mitigation
- **LLM-Driven Agents** — 4 supply chain roles each powered by an LLM that reasons about inventory, backorders, costs, demand trends, and constraint compliance
- **AI Coordinator** — Global supply chain analysis with per-role order quantity recommendations and bullwhip severity assessment
- **22+ Demand Pattern Types** — Step, random, seasonal, normal, Poisson, ARMA, jump diffusion, regime switching, volatility clustering, promotion-driven, and more
- **18 LLM Provider Presets** — OpenAI (GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo), Anthropic (Claude Opus/Sonnet/Haiku 4), DeepSeek (V3, R1), Zhipu GLM, Moonshot Kimi, Qwen, OpenRouter, Groq, Together AI, Ollama (local), custom OpenAI-compatible, plus Mock client
- **Interactive 3D Visualization** — Real-time and playback 3D views with animated logistics flows, inventory panels, and supply chain connection lines
- **Bullwhip Effect Analysis** — Coefficient of variation, amplification ratios, and variance ratios per tier; comparison across constraint configurations
- **Dual Interface** — Streamlit web UI for interactive experimentation + CLI for batch runs and comparison studies
- **Decision Analysis** — Overreaction detection, order distribution visualization, and per-agent LLM reasoning inspection
- **Result Export** — JSON, Excel, interactive HTML dashboards, and PNG charts

## Architecture

```
Customer Demand → Retailer → Wholesaler → Distributor → Manufacturer
                    ↑          ↑            ↑              ↑
                    ├── Material / Product Flow ──────────→
                    ←── Order Information Flow ────────────
                              ↗
                         AI Coordinator
                    (Global analysis & guidance)

              ┌─── Order Constraint Layers ───┐
              │  1. Fixed Min/Max Bounds      │
              │  2. Discrete Point Selection  │
              │  3. Adaptive Dynamic Limits   │
              └───────────────────────────────┘
```

Each round:
1. Customer demand is generated based on the selected demand pattern
2. Adaptive limits are recomputed based on latest demand history (if enabled)
3. Each agent receives its order constraints in the LLM prompt, makes a decision, and the result is clamped through the constraint pipeline
4. Shipments are processed downstream, orders placed upstream
5. The Coordinator analyzes all positions and issues per-role guidance
6. Costs (holding + backorder) are calculated for each role

## Quick Start

### Prerequisites

- **Python 3.10+**
- **pip**

### Installation

```bash
git clone https://github.com/guxuan1210/llm-beer-game.git
cd llm-beer-game
pip install -r requirements.txt
```

### Launch Web UI (Recommended)

```bash
python run_web_ui.py
```

Or directly with Streamlit:

```bash
streamlit run llm_beer_game/ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser. In the sidebar:
1. Select a **Config Template** (Default, Information Sharing, Rule-Based, or Custom)
2. Choose a **Demand Pattern** and set simulation weeks
3. **Enable order constraints** — the core feature:
   - Check "Enable Min/Max Order Quantity Limit" for fixed bounds per role
   - Check "Enable Discrete Point Ordering" for discrete selection
   - Check "Enable Adaptive Order Quantity Limit" for dynamic bounds
4. Select an **LLM Provider** (or use Mock for offline testing)
5. Click **Play** to start the simulation

The web UI includes these tabs:
- **3D Supply Chain** — Real-time 3D visualization of the running simulation
- **3D Supply Chain (Playback)** — Play back completed simulations at adjustable speed (1x–20x)
- **Data Table** — Per-round inventory, orders, shipments, and costs for each role
- **2D Charts** — Interactive Plotly charts for inventory, orders, bullwhip effect, and costs
- **Decision Statistics** — LLM decision reasoning, order distribution, and agent comparison
- **Coordinator Analysis** — Coordinator mind maps and per-round guidance
- **Prompt Debug** — Inspect and edit the prompts sent to each LLM agent

> **Note**: By default, the simulation uses the **Mock LLM client** (no API key needed). Select a different provider in the sidebar to use a real LLM.

## LLM Provider Setup

### Mock (Default — No API Key Required)
The Mock client uses rule-based logic internally. Select "Mock LLM (Testing)" in the UI sidebar. Ideal for testing the simulation flow and order constraint configurations without API costs.

### OpenAI
1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. In the Streamlit UI sidebar, select an OpenAI preset (GPT-4o, GPT-4o-mini, etc.)
3. Enter your API key in the "API Key" field
4. Or set the environment variable: `export OPENAI_API_KEY="sk-..."`

### Anthropic (Claude)
1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Select "Claude Opus 4", "Claude Sonnet 4", or "Claude Haiku 4"
3. Enter your API key in the sidebar
4. Or: `export ANTHROPIC_API_KEY="sk-ant-..."`

### Ollama (Local Models, No API Key)
1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3:70b
   ollama pull gemma3:27b
   ollama pull qwen2:72b
   ```
2. In the UI, select an Ollama preset (Gemma 3, Llama 3, Qwen 2)
3. Default base URL: `http://localhost:11434` — adjust if your Ollama runs on a different port

### Other Cloud Providers
Configure via the **Custom config** template or the "Custom OpenAI-Compatible API" preset. Providers like DeepSeek, Zhipu, Qwen, and Moonshot have their own API key systems — obtain keys from their respective platforms and enter them in the UI sidebar.

### API Key Storage
API keys are stored **per-session** in the Streamlit session state. They can also be saved to JSON config files via the config manager. No `.env` file is required.

## Configuration

### Config Templates

| Template | Description |
|---|---|
| **Custom** | Full control over all parameters including order constraints |
| **Default Config** | Step demand (jumps from 1→8 at week 5), LLM agents, 50 weeks |
| **Information Sharing** | Same as default but with demand/inventory visibility shared across roles |
| **Rule-Based** | All agents use base-stock policies (no LLM), good for benchmarking constraint effectiveness |

### Key Simulation Parameters

| Parameter | Default | Description |
|---|---|---|
| `total_weeks` | 50 | Number of simulation rounds |
| `lead_time` | 2 | Order fulfillment delay (weeks) |
| `holding_cost` | 0.5 | Cost per unit held in inventory per week |
| `backorder_cost` | 1.0 | Cost per unit backordered per week |
| `random_seed` | Optional | Fixed seed for reproducible results |
| `information_sharing` | false | Whether agents see upstream/downstream data |

### Agent Initial Conditions

| Role | Initial Inventory | Initial In-Transit |
|---|---|---|
| Retailer | 12 | [4, 4] |
| Wholesaler | 12 | [4, 4] |
| Distributor | 12 | [4, 4] |
| Manufacturer | 12 | [4, 4] |

## Demand Patterns

The framework supports **22+ demand pattern types** organized into 5 categories:

| Category | Patterns |
|---|---|
| **Basic** | Step, Random (Uniform), Seasonal, Custom |
| **Common Distributions** | Normal, Poisson, Exponential, Triangular, Beta, Log-normal |
| **Mixed Distributions** | Bimodal, Seasonal+Random, Trend+Cyclic, Multi-stage, Markov Chain |
| **Real Market Scenarios** | Promotion-driven, Competition-affected, New Product Diffusion, Inventory-sensitive |
| **Unstable / Advanced** | Autoregressive (AR), ARMA, Jump Diffusion, Poisson Jump, Regime Switching, Volatility Clustering |

Each pattern has configurable parameters (mean, std, jump magnitude, regime probabilities, etc.) exposed in the Streamlit UI and `DemandConfig` dataclass. Testing order constraint effectiveness across different demand patterns is a key experimental use case.

## CLI Usage

The CLI is useful for batch runs, comparison studies across constraint configurations, and integration into automated workflows.

```bash
# Run a single simulation with default config
python -m llm_beer_game.main run

# Run with a named config
python -m llm_beer_game.main run --config info_sharing

# Run with a JSON config file and custom output directory
python -m llm_beer_game.main run --config deepseek_chat --output results/

# Run a 3-scenario comparison study (Standard vs Info Sharing vs Rule-Based)
python -m llm_beer_game.main compare --output comparison_results/

# Generate sample config files
python -m llm_beer_game.main config

# Launch the Streamlit web UI from CLI
python -m llm_beer_game.main web

# Verbose output
python -m llm_beer_game.main run --verbose
```

## Supported LLM Providers

| Provider | Presets | Model | Base URL | API Key |
|---|---|---|---|---|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | `https://api.openai.com/v1` | Required |
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 4 | claude-opus-4, claude-sonnet-4, claude-haiku-4 | `https://api.anthropic.com` | Required |
| **DeepSeek** | DeepSeek Chat (V3), Reasoner (R1) | deepseek-chat, deepseek-reasoner | `https://api.deepseek.com` | Required |
| **Zhipu** | GLM-4, GLM-4 Flash | glm-4, glm-4-flash | `https://open.bigmodel.cn/api/paas/v4` | Required |
| **Moonshot** | Kimi | moonshot-v1-8k | `https://api.moonshot.cn/v1` | Required |
| **Qwen (DashScope)** | Qwen Max, Qwen Plus | qwen-max, qwen-plus | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Required |
| **OpenRouter** | Auto Route | openai/gpt-4o (routable) | `https://openrouter.ai/api/v1` | Required |
| **Groq** | Llama 3 70B | llama3-70b-8192 | `https://api.groq.com/openai/v1` | Required |
| **Together AI** | Llama 3 70B | meta-llama/Llama-3-70b-chat-hf | `https://api.together.xyz/v1` | Required |
| **Ollama** | Gemma 3 27B, Llama 3 70B, Qwen 2 72B | gemma3:27b, llama3:70b, qwen2:72b | `http://localhost:11434` | Not required |
| **GPT-OSS** | GPT-OSS 20B (Local) | gpt-oss:20b | `http://localhost:11434` | Not required |
| **Custom** | OpenAI-Compatible API | User-specified | User-specified | Optional |
| **Mock** | Mock LLM (Testing) | mock-llm | N/A | Not required |

## Project Structure

```
llm-beer-game/
├── README.md
├── requirements.txt              # Python dependencies
├── run_web_ui.py                 # Streamlit launcher script
│
├── configs/                      # Pre-built LLM config JSON files
│   ├── deepseek_chat.json
│   ├── gpt_oss_20b.json
│   └── llm_custom_presets.json
│
├── examples/                     # Example scripts
│   ├── basic_simulation.py
│   ├── ollama_configs.py
│   ├── ollama_simulation.py
│   └── test_ollama.py
│
├── results/                      # Simulation output directory
│
├── llm_beer_game/                # Main Python package
│   ├── __init__.py               # Package root — exports public API
│   ├── main.py                   # CLI entry point
│   │
│   ├── config/                   # Configuration management
│   │   ├── game_config.py        # GameConfig, AgentConfig, DemandConfig,
│   │   │                         # LLMConfig, order constraint fields,
│   │   │                         # 18 built-in LLM provider presets
│   │   └── adaptive_limits_config.py  # AdaptiveLimitsConfig dataclass
│   │
│   ├── agents/                   # Agent implementations
│   │   ├── base_agent.py         # Base agent class
│   │   ├── llm_agent.py          # LLM-powered agent + order constraint enforcement
│   │   ├── adaptive_agent.py     # AdaptiveLLMAgent, AdaptiveRuleBasedAgent
│   │   ├── adaptive_limits.py    # AdaptiveLimitsCalculator (ratio/window/forecast)
│   │   ├── coordinator_agent.py  # AI coordinator for global analysis
│   │   ├── mindmap_decision_agent.py
│   │   ├── mindmap_templates.py
│   │   └── supply_chain_agents.py # Role-specific agents with prompt-level constraints
│   │
│   ├── llm/                      # LLM client integrations
│   │   └── llm_client.py         # OpenAI, Anthropic, Ollama, Mock clients
│   │
│   ├── simulation/               # Simulation engine
│   │   └── game_engine.py        # GameEngine, SimulationResult, DemandPattern
│   │
│   ├── analysis/                 # Analysis tools
│   │   ├── bullwhip_analyzer.py  # Bullwhip effect metrics
│   │   ├── decision_analyzer.py  # LLM decision analysis
│   │   └── overreaction_analyzer.py
│   │
│   ├── visualization/            # Visualization
│   │   ├── visualizer.py         # 2D charts (matplotlib + plotly)
│   │   ├── supply_chain_3d.py    # 3D playback visualization (Three.js)
│   │   └── realtime_3d.py        # 3D real-time visualization (Three.js)
│   │
│   ├── ui/                       # Web UI
│   │   └── streamlit_app.py      # Streamlit application with order constraint controls
│   │
│   └── utils/                    # Utility functions
│       ├── helpers.py
│       └── ollama_utils.py
│
└── demo_comparison/              # Pre-computed comparison results
```

## Analysis & Visualization

### 3D Supply Chain Visualization

The 3D view shows:
- **4 agent buildings** (Retailer, Wholesaler, Distributor, Manufacturer) on a semi-transparent platform
- **Customer** position showing incoming market demand
- **Color-coded ground arrows** indicating the flow direction between nodes
- **Animated shipment flows** — brown boxes moving between agents showing goods in transit
- **Pulsing connection lines** highlighting active order/shipment paths
- **Per-node floating info panels** showing inventory, orders, shipments, and costs
- **OrbitControls** — drag to rotate, scroll to zoom, right-drag to pan

### Bullwhip Analysis

- **Coefficient of Variation** — Order variability relative to the mean for each tier
- **Amplification Ratios** — CV(tierₙ) / CV(tierₙ₋₁), measuring how variability amplifies upstream
- **Overall Bullwhip Effect** — Composite metric across the entire chain
- **Constraint Impact Comparison** — Compare bullwhip metrics with and without order constraints enabled

### Decision Statistics

- Order quantity distribution per agent (histograms)
- Reasoning summary extraction from LLM responses
- Overreaction detection (sudden large order spikes vs. constraint-compliant behavior)
- Constraint violation tracking (how often orders hit min/max bounds)

## References

- **Beer Distribution Game** — Originally developed at MIT Sloan School of Management as a teaching tool for supply chain dynamics
- **Bullwhip Effect** — Lee, Padmanabhan, and Whang (1997). "Information Distortion in a Supply Chain: The Bullwhip Effect"
- Research papers included in the repository (PDFs):
  - LLM-driven Beer Distribution Game design document (Chinese)
  - Xiaotian Liu (2022) — Related supply chain modeling
  - Oroojlooyjadid et al. (2021) — Deep reinforcement learning for beer game
  - Stochastic DRO — Distributionally robust optimization in supply chains

## License

This project is provided for research and educational purposes.
