# LLM Beer Game — Multi-Agent Supply Chain Simulation

A Python-based Beer Distribution Game where **LLM-powered agents** (Retailer → Wholesaler → Distributor → Manufacturer) make autonomous ordering decisions. Supports **12+ LLM providers** including local models via Ollama, with a **three-layer order constraint system** to mitigate the bullwhip effect.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web UI
python run_web_ui.py
# Or: streamlit run llm_beer_game/ui/streamlit_app.py
```

Open **http://localhost:8501**. Default configuration uses **Ollama with gemma3:27b** on `http://localhost:11434`. For offline testing, switch to Mock LLM in the sidebar.

## Architecture

```
Customer Demand → Retailer → Wholesaler → Distributor → Manufacturer
                    ↑          ↑            ↑              ↑
                    ←── Order Information Flow ────────────
                    ──→ Material / Product Flow ──────────→

              ┌─── Order Constraint Layers ───┐
              │  1. Fixed Min/Max Bounds      │
              │  2. Discrete Point Selection  │
              │  3. Adaptive Dynamic Limits   │
              └───────────────────────────────┘
```

**Each simulation round:**
1. Customer demand is generated based on the selected pattern
2. Adaptive limits are recomputed from latest demand history (if enabled)
3. Each agent receives its order constraints in the LLM prompt, reasons about inventory/backorders/costs, and makes a decision
4. The decision is validated through the constraint pipeline and clamped if necessary
5. Shipments are processed downstream, orders placed upstream
6. Costs (holding + backorder) are calculated for each role

## Key Features

- **LLM-Driven Agents** — 4 supply chain roles (Retailer, Wholesaler, Distributor, Manufacturer) each powered by an LLM that reasons about inventory, backorders, demand trends, and costs
- **Per-Role Decision Constraint Control** — Each role can individually disable order constraints, removing all limit/constraint language from prompts for unrestricted decision-making
- **Three-Layer Order Constraint System** — Fixed min/max bounds, discrete point selection, and adaptive dynamic limits (ratio/window/forecast algorithms)
- **Combined Inventory & Demand Chart** — All 4 roles' net inventory overlaid with customer demand on a single chart for holistic supply chain visibility
- **22+ Demand Patterns** — Step, random, seasonal, normal, Poisson, ARMA, jump diffusion, regime switching, volatility clustering, promotion-driven, and more
- **12 LLM Providers** — OpenAI, Anthropic, Ollama (local), DeepSeek, Zhipu GLM, Moonshot Kimi, Qwen, OpenRouter, Groq, Together AI, GPT-OSS, Custom OpenAI-compatible, plus Mock for offline testing
- **Demand Forecasting Add-on** — Optional checkbox to append structured demand forecasting deduction rules to agent prompts without modifying original prompt text
- **AI Coordinator** — Global supply chain analysis with per-role guidance and bullwhip diagnosis (optional, disabled by default)
- **Information Sharing Mode** — Agents share inventory, demand, and pipeline data across the chain
- **3D Visualization** — Real-time and playback 3D views with animated logistics flows, inventory panels, and multi-CDN fallback for restricted network environments
- **Bullwhip Analysis** — Coefficient of variation, amplification ratios, and variance ratios per tier
- **Decision Analysis** — Overreaction detection, order distribution visualization, LLM reasoning inspection
- **Result Export** — JSON, Excel, HTML dashboards, PNG charts

## LLM Provider Setup

| Provider | Preset Models | Base URL | API Key |
|---|---|---|---|
| **Ollama** (default) | gemma3:27b, llama3:70b, qwen2:72b | `http://localhost:11434` | Not required |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | `https://api.openai.com/v1` | Required |
| **Anthropic** | claude-opus-4, claude-sonnet-4, claude-haiku-4 | — | Required |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | `https://api.deepseek.com` | Required |
| **Zhipu GLM** | glm-4, glm-4-flash | `https://open.bigmodel.cn/api/paas/v4` | Required |
| **Moonshot Kimi** | moonshot-v1-8k | `https://api.moonshot.cn/v1` | Required |
| **Qwen** | qwen-max, qwen-plus | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Required |
| **OpenRouter** | openai/gpt-4o (routable) | `https://openrouter.ai/api/v1` | Required |
| **Groq** | llama3-70b-8192 | `https://api.groq.com/openai/v1` | Required |
| **Together AI** | meta-llama/Llama-3-70b-chat-hf | `https://api.together.xyz/v1` | Required |
| **GPT-OSS** | gpt-oss:20b | `http://localhost:11434` | Not required |
| **Custom** | User-specified | User-specified | Optional |
| **Mock** | mock-llm | — | Not required |

The **Base URL auto-updates** when switching providers in the UI. All OpenAI-compatible providers (DeepSeek, Zhipu, Moonshot, Qwen, OpenRouter, Groq, Together) share the same API format and are routed through the `OpenAIClient` with their respective endpoints.

### Ollama (Local — Default)

```bash
ollama pull gemma3:27b
```

The UI dynamically fetches installed models from the Ollama API and displays them in a dropdown. If the service is unreachable, a fallback list of recommended models is shown.

### Mock (Offline Testing)

Select "Mock LLM" in the provider dropdown. Uses rule-based heuristics internally — no API calls, no network needed. Ideal for testing simulation flow and constraint configurations.

## Order Constraint System

### Layer 1: Fixed Min/Max Bounds
Per-role hard floor and ceiling on order quantities. Agents are instructed about limits in both system and user prompts. Orders outside range are clamped.

### Layer 2: Discrete Point Selection
Restricts orders to a predefined set of allowed values (e.g., {5, 10, 15, 20}). LLM output is snapped to the nearest valid point.

### Layer 3: Adaptive Dynamic Limits
Dynamically recomputes [min, max] each round based on actual retailer demand. Three algorithms:

| Method | Behavior | Best For |
|---|---|---|
| `ratio` | Scales base limits by demand/average ratio | Stable demand |
| `window` | Sets limits via mean ± α×σ over sliding window | Volatile demand |
| `forecast` | Moving average + trend with fixed multipliers | Trending demand |

Upstream roles get progressively wider bands: Retailer ×1.0 → Wholesaler ×1.1 → Distributor ×1.2 → Manufacturer ×1.3.

### Per-Role "No Decision Constraints"

Each role has an individual **"No Decision Constraints"** checkbox in the UI sidebar (under "Order Quantity Limit"). When enabled for a role:

- All order limit/min/max/discrete point language is **removed** from that role's system and user prompts
- The "Resolving Horizon Conflicts" section replaces "order MAX/MIN" with "order aggressively/conservatively" (or "produce aggressively/conservatively" for Manufacturer)
- Correct examples become neutral (no constraint range references)
- `_apply_order_limits()` still clamps to `[0, 1000]` as an invisible safety net

This allows studying how each agent behaves when given complete decision freedom, and testing whether constraints or freedom at specific supply chain tiers amplifies or dampens the bullwhip effect.

## Demand Forecasting Add-on

A checkbox in the UI: **"Enable Supply Chain Demand Forecasting Deduction"**. When checked, a structured forecasting fragment is appended to the end of each agent's system prompt — without modifying any existing prompt text. Agents are instructed to:

- Identify their supply chain role
- Predict downstream and full-chain demand trends (short-term and mid-term)
- Structurally record forecast rationale, change direction, affected nodes, and potential trajectories
- Maintain a dedicated forecast block and iteratively revise it each round
- Output forecasts as an add-on without altering the original response structure

When unchecked, prompts are completely unchanged.

## Demand Patterns

22+ pattern types across 5 categories:

| Category | Patterns |
|---|---|
| **Basic** | Step, Random (Uniform), Seasonal, Custom |
| **Distributions** | Normal, Poisson, Exponential, Triangular, Beta, Log-normal |
| **Mixed** | Bimodal, Seasonal+Random, Trend+Cyclic, Multi-stage, Markov Chain |
| **Real Market** | Promotion-driven, Competition-affected, Diffusion, Inventory-sensitive |
| **Advanced** | AR, ARMA, Jump Diffusion, Poisson Jump, Regime Switching, Volatility Clustering |

Parameters (mean, std, amplitude, period, jump magnitude, regime probabilities, etc.) are configurable via the UI sidebar.

## Configuration

### Default Simulation Parameters

| Parameter | Default |
|---|---|
| Total weeks | 50 |
| Order lead time | 2 weeks (all roles) |
| Transport lead time | 2 weeks (all roles except Manufacturer: 1 week production) |
| Holding cost | 0.5 / unit / week |
| Backorder cost | 1.0 / unit / week |
| Initial inventory | 12 (all roles) |
| Initial in-transit | [4, 4] (all roles) |
| LLM max tokens | 150 |
| LLM temperature | 0.7 |

### Agent Costs

| Role | Holding Cost | Backorder Cost |
|---|---|---|
| Retailer | 0.5 | 1.0 |
| Wholesaler | 0.5 | 1.0 |
| Distributor | 0.5 | 1.0 |
| Manufacturer | 0.5 | 1.0 |

Costs are configurable per role in the UI sidebar.

## CLI Usage

```bash
# Run a single simulation
python -m llm_beer_game.main run

# Run with named config
python -m llm_beer_game.main run --config info_sharing

# Comparison study (Standard vs Info Sharing vs Rule-Based)
python -m llm_beer_game.main compare --output comparison_results/

# Generate sample config files
python -m llm_beer_game.main config

# Launch web UI from CLI
python -m llm_beer_game.main web
```

## Project Structure

```
llm-beer-game/
├── README.md
├── requirements.txt
├── run_web_ui.py                      # Streamlit launcher
│
├── configs/                           # Pre-built config files
│   ├── deepseek_chat.json
│   ├── gpt_oss_20b.json
│   └── llm_custom_presets.json
│
├── examples/                          # Example scripts
│   ├── basic_simulation.py
│   ├── ollama_configs.py
│   ├── ollama_simulation.py
│   └── test_ollama.py
│
└── llm_beer_game/                     # Main package
    ├── __init__.py
    ├── main.py                        # CLI entry point
    │
    ├── config/
    │   ├── game_config.py             # GameConfig, AgentConfig, DemandConfig,
    │   │                              # LLMConfig, SimulationConfig, CostConfig,
    │   │                              # 18 built-in LLM provider presets
    │   └── adaptive_limits_config.py  # AdaptiveLimitsConfig
    │
    ├── agents/
    │   ├── base_agent.py              # BaseAgent (ABC), AgentState
    │   ├── llm_agent.py               # LLMAgent — prompt building, JSON parsing,
    │   │                              # order limit enforcement, LLM decision loop
    │   ├── supply_chain_agents.py     # Retailer, Wholesaler, Distributor,
    │   │                              # Manufacturer agent subclasses
    │   ├── coordinator_agent.py       # CoordinatorAgent — global analysis & guidance
    │   ├── adaptive_agent.py          # AdaptiveLLMAgent, AdaptiveRuleBasedAgent
    │   ├── adaptive_limits.py         # AdaptiveLimitsCalculator (ratio/window/forecast)
    │   ├── mindmap_decision_agent.py  # Mindmap-based decision with shared info fusion
    │   └── mindmap_templates.py       # Role-specific mind map templates
    │
    ├── llm/
    │   └── llm_client.py              # OpenAIClient, AnthropicClient, OllamaClient,
    │                                  # GPTOSSClient, MockLLMClient, LLMManager
    │
    ├── simulation/
    │   └── game_engine.py             # GameEngine, SimulationResult, 22+ demand generators
    │
    ├── analysis/
    │   ├── bullwhip_analyzer.py       # Bullwhip effect metrics & comparison
    │   ├── decision_analyzer.py       # Decision quality scoring & pattern detection
    │   └── overreaction_analyzer.py   # Overreaction detection & severity classification
    │
    ├── visualization/
    │   ├── visualizer.py              # 2D charts (matplotlib + plotly)
    │   ├── supply_chain_3d.py         # 3D playback (Three.js, multi-CDN fallback)
    │   └── realtime_3d.py             # 3D real-time visualization
    │
    ├── ui/
    │   └── streamlit_app.py           # Full Streamlit web application
    │
    └── utils/
        ├── helpers.py                 # Logging, timers, validators, exporters
        └── ollama_utils.py            # Ollama model discovery & status checks
```

## Web UI Tabs

- **3D Supply Chain (Live)** — Real-time 3D visualization during simulation
- **3D Supply Chain (Playback)** — Play back completed simulations at adjustable speed
- **Data Table** — Per-round inventory, orders, shipments, and costs
- **2D Charts** — Interactive Plotly charts for inventory (per-role + combined view with demand overlay), orders, shipments, in-transit, bullwhip, costs
- **Decision Statistics** — LLM reasoning, order distribution, agent comparison
- **Prompt Debug** — Inspect system/user prompts sent to each agent

## References

- **Beer Distribution Game** — Originally developed at MIT Sloan School of Management
- Lee, Padmanabhan, and Whang (1997). "Information Distortion in a Supply Chain: The Bullwhip Effect"
- Oroojlooyjadid et al. (2021) — Deep reinforcement learning for beer game
