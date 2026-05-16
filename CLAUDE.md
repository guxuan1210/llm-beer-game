# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LLM Beer Game — a Python multi-agent supply chain simulation where LLM-powered agents (Retailer → Wholesaler → Distributor → Manufacturer) make autonomous ordering decisions. Models the classic MIT Beer Distribution Game to study the bullwhip effect with AI agents.

## Common commands

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit web UI (primary interface)
python run_web_ui.py

# CLI: run a single simulation
python -m llm_beer_game.main run

# CLI: run with named config
python -m llm_beer_game.main run --config info_sharing

# CLI: comparison study (Standard vs Info Sharing vs Rule-Based)
python -m llm_beer_game.main compare --output comparison_results/

# CLI: generate sample config files
python -m llm_beer_game.main config

# Run example
python examples/basic_simulation.py

# Lint
flake8 llm_beer_game/
# Format
black llm_beer_game/
# Sort imports
isort llm_beer_game/
# Type check
mypy llm_beer_game/
```

No test suite exists yet despite pytest being in requirements.txt. The `examples/` directory serves as the primary verification mechanism.

## Architecture

### Simulation loop (`GameEngine.run_simulation`)

Each round executes downstream→upstream for each of the 4 roles:

1. `generate_demand(round)` — pick from 22+ demand patterns
2. `update_shared_information()` — if info sharing enabled, collect state from all agents and distribute
3. `agent.process_round(demand)` — each agent: receive demand → receive shipment → fulfill → **make LLM decision** → place order → calculate costs
4. The downstream agent's order becomes the upstream agent's demand

### Agent decision pipeline

`BaseAgent` ([llm_beer_game/agents/base_agent.py](llm_beer_game/agents/base_agent.py)) defines the lifecycle. `LLMAgent` ([llm_beer_game/agents/llm_agent.py](llm_beer_game/agents/llm_agent.py)) adds:

- **`_create_system_prompt()`** — role identity, dual-horizon framework (short-term net position + long-term coverage ratio over lead time), cost tradeoff (backorder/holding = 2:1), JSON output format
- **`_create_user_prompt(context)`** — current state, pipeline breakdown per period, short/long-term balance analysis with explicit guidance (MIN/MAX suggestions), historical trends, cost analysis, shared info if enabled, order constraints
- **`make_decision()`** → LLM call → `_parse_llm_response()` — extracts JSON via regex, falls back to number extraction, then to heuristic fallback
- **`_apply_order_limits()`** — three-layer constraint: discrete point snapping → min/max clamping
- Four role subclasses in [llm_beer_game/agents/supply_chain_agents.py](llm_beer_game/agents/supply_chain_agents.py) override `_create_system_prompt()` with role-specific guidance

Agent state is tracked in `AgentState` dataclass: inventory, backorder, shipment_pipeline (FIFO queue of length = effective lead_time), history arrays.

### Lead time model

Each agent has three components: `order_lead_time` + `transport_lead_time` + `production_lead_time`. Manufacturer has only production_lead_time; others have order + transport. Effective `lead_time` = sum of all three (min 1). This determines the `shipment_pipeline` length.

### LLM client abstraction

`LLMClient` ABC in [llm_beer_game/llm/llm_client.py](llm_beer_game/llm/llm_client.py). Implementations:
- `OpenAIClient` — handles all OpenAI-compatible providers (OpenAI, DeepSeek, Zhipu, Moonshot, Qwen, OpenRouter, Groq, Together) via `LLMClientFactory.PROVIDER_ALIASES`
- `AnthropicClient` — supports extended thinking via `enable_thinking(budget_tokens)`
- `OllamaClient` — local models via `/api/generate`
- `GPTOSSClient` — Harmony format via `/api/chat`
- `MockLLMClient` — heuristic rules based on inventory/demand/backorder extraction from prompt
- `LLMManager` — multi-client failover with retry

### Three-layer order constraint system

1. **Fixed bounds** — `AgentConfig.min_order_quantity` / `max_order_quantity`, clamped in `_apply_order_limits()`
2. **Discrete points** — `AgentConfig.enable_discrete_points` + `discrete_order_points`, LLM output snapped to nearest valid point
3. **Adaptive dynamic limits** — `AdaptiveLimitsConfig` with ratio/window/forecast algorithms, recomputed each round from retailer demand history, upstream roles get progressively wider bands

### Configuration

`GameConfig` dataclass ([llm_beer_game/config/game_config.py](llm_beer_game/config/game_config.py)) composes: `SimulationConfig`, `AgentConfig` ×4, `DemandConfig`, `LLMConfig`, `CostConfig`, `AdaptiveLimitsConfig`. 18 built-in `LLMProviderPreset`s in `BUILTIN_LLM_PRESETS`. `ConfigManager` handles JSON/YAML serialization of configs to `configs/`.

### Information sharing

When `SimulationConfig.information_sharing = True`, each round all agents' state (inventory, backorder, demand, orders, pipeline) is collected and distributed via `set_shared_info()`. The LLM prompt then includes a global supply-demand balance analysis section showing upstream/downstream states, chain-wide order totals vs retailer demand, and amplification warnings.

### Demand forecasting add-on

Controlled by `GameConfig.enable_demand_forecasting`. When enabled, appends a structured forecasting instruction block to the system prompt without modifying existing prompt text. Agents are told to maintain a dedicated forecast block and iteratively revise each round.

### Web UI

Streamlit app at [llm_beer_game/ui/streamlit_app.py](llm_beer_game/ui/streamlit_app.py). Six tabs: 3D Live, 3D Playback, Data Table, 2D Charts, Decision Statistics, Prompt Debug. Sidebar configures LLM provider/model, demand pattern/params, order constraints, costs, simulation weeks. The 3D visualization uses Three.js with multi-CDN fallback for restricted networks.

### Key architectural patterns

- **`sys.path` manipulation**: Many files use `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` for intra-package imports instead of relative imports. The package `__init__.py` re-exports all public API.
- **Prompt engineering is the core logic**: Agent behavior is primarily driven by prompt content, not code logic. Changes to agent decision-making should focus on prompt modifications.
- **JSON parsing is defensive**: The `_parse_llm_response` method has multiple fallback strategies (JSON regex extraction → standalone number extraction → keyword-based extraction → heuristic fallback) because LLM output format is unreliable.
- **State lives in agents, not engine**: Agent state (inventory, pipelines, history) is owned by agent instances. The engine orchestrates but doesn't hold business state.
