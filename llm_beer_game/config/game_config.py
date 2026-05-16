"""Game Configuration Module

Defines game parameters, scenario configurations, and LLM settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import yaml
from pathlib import Path


class AgentType(Enum):
    """Agent type enumeration"""
    LLM = "llm"
    RULE_BASED = "rule_based"
    HUMAN = "human"


class DemandPatternType(Enum):
    """Demand pattern type"""
    # Basic distribution types
    STEP = "step"
    RANDOM = "random"  # Uniform distribution
    SEASONAL = "seasonal"
    CUSTOM = "custom"

    # Common distribution types
    NORMAL = "normal"  # Normal distribution
    POISSON = "poisson"  # Poisson distribution
    EXPONENTIAL = "exponential"  # Exponential distribution
    TRIANGULAR = "triangular"  # Triangular distribution
    BETA = "beta"  # Beta distribution
    LOGNORMAL = "lognormal"  # Log-normal distribution

    # Mixed distribution types
    BIMODAL = "bimodal"  # Bimodal distribution
    SEASONAL_RANDOM = "seasonal_random"  # Seasonal + random mix
    TREND_CYCLIC = "trend_cyclic"  # Trend + cyclical mix
    MULTISTAGE = "multistage"  # Multi-stage mixed demand
    MARKOV = "markov"  # Markov chain demand

    # Real market scenario types
    PROMOTION = "promotion"  # Promotion-driven demand
    COMPETITION = "competition"  # Competition-affected demand
    DIFFUSION = "diffusion"  # New product diffusion demand
    INVENTORY_SENSITIVE = "inventory_sensitive"  # Inventory-sensitive demand

    # Unstable demand types
    AUTOREGRESSIVE = "autoregressive"  # Autoregressive demand (AR)
    ARMA = "arma"  # ARMA demand
    JUMP_DIFFUSION = "jump_diffusion"  # Jump diffusion demand
    POISSON_JUMP = "poisson_jump"  # Poisson jump demand
    REGIME_SWITCHING = "regime_switching"  # Regime switching demand
    VOLATILITY_CLUSTERING = "volatility_clustering"  # Volatility clustering demand


@dataclass
class CostConfig:
    """Cost configuration"""
    holding_cost: float = 0.5  # Inventory holding cost
    backorder_cost: float = 1.0  # Backorder cost
    order_cost: float = 0.0  # Order cost


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_type: AgentType = AgentType.LLM
    initial_inventory: int = 12
    initial_backorder: int = 0
    # Initial in-transit pipeline (list of arrival quantities per period), shortfall padded to effective lead time with 0
    initial_in_transit: Optional[List[int]] = None
    cost_config: CostConfig = field(default_factory=CostConfig)

    # Order quantity limit parameters
    min_order_quantity: Optional[int] = None  # Minimum order quantity
    max_order_quantity: Optional[int] = None  # Maximum order quantity

    # Discrete point selection parameters
    enable_discrete_points: bool = False  # Enable discrete point selection
    discrete_order_points: Optional[List[int]] = None  # List of discrete order points

    # When True, all order limit/constraint language is removed from this role's prompts.
    # The agent makes unrestricted decisions; _apply_order_limits still clamps to [0, 1000].
    no_decision_constraints: bool = False

    # Rule-based agent specific parameters
    base_stock_level: Optional[int] = None
    ma_window: Optional[int] = None

    # LLM agent specific parameters
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150


@dataclass
class DemandConfig:
    """Demand configuration"""
    pattern_type: DemandPatternType = DemandPatternType.STEP
    base_demand: int = 4

    # Step demand parameters
    step_week: int = 5
    step_demand: int = 8
    step_duration: int = 4

    # Random demand parameters (uniform distribution)
    random_min: int = 1
    random_max: int = 10
    random_seed: Optional[int] = None

    # Seasonal demand parameters
    seasonal_amplitude: float = 2.0
    seasonal_period: int = 12

    # Custom demand sequence
    custom_demands: List[int] = field(default_factory=list)

    # Normal distribution parameters
    normal_mean: float = 10.0
    normal_std: float = 2.0

    # Poisson distribution parameters
    poisson_lambda: float = 8.0

    # Exponential distribution parameters
    exponential_lambda: float = 0.1
    exponential_base: float = 5.0  # Base demand value

    # Triangular distribution parameters
    triangular_min: float = 1.0
    triangular_max: float = 20.0
    triangular_mode: float = 10.0  # Mode

    # Beta distribution parameters
    beta_alpha: float = 2.0
    beta_beta: float = 5.0
    beta_scale: float = 20.0  # Scale factor
    beta_shift: float = 1.0   # Shift

    # Log-normal distribution parameters
    lognormal_mu: float = 2.0
    lognormal_sigma: float = 0.5

    # Bimodal distribution parameters
    bimodal_mean1: float = 5.0
    bimodal_std1: float = 1.0
    bimodal_mean2: float = 15.0
    bimodal_std2: float = 2.0
    bimodal_weight: float = 0.6  # Weight of first peak

    # Seasonal + random mix parameters
    seasonal_random_base: float = 10.0
    seasonal_random_amplitude: float = 3.0
    seasonal_random_period: int = 12
    seasonal_random_noise_std: float = 1.5

    # Trend + cyclical mix parameters
    trend_cyclic_base: float = 8.0
    trend_cyclic_slope: float = 0.1  # Trend slope
    trend_cyclic_amplitude: float = 2.0
    trend_cyclic_period: int = 8

    # Multi-stage mixed demand parameters
    multistage_stages: List[Dict] = field(default_factory=lambda: [
        {"duration": 10, "type": "normal", "mean": 5, "std": 1},
        {"duration": 15, "type": "normal", "mean": 12, "std": 2},
        {"duration": 10, "type": "exponential", "lambda": 0.2, "base": 8}
    ])

    # Markov chain demand parameters
    markov_states: List[float] = field(default_factory=lambda: [3, 8, 15])
    markov_transition_matrix: List[List[float]] = field(default_factory=lambda: [
        [0.7, 0.2, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])
    markov_initial_state: int = 1

    # Promotion-driven demand parameters
    promotion_base_demand: float = 8.0
    promotion_intensity: float = 3.0  # Promotion intensity multiplier
    promotion_start_week: int = 10
    promotion_duration: int = 3
    promotion_decay_rate: float = 0.5  # Post-promotion decay rate

    # Competition-affected demand parameters
    competition_base_demand: float = 10.0
    competition_market_share: float = 0.4
    competition_elasticity: float = 1.5
    competition_competitor_actions: List[Dict] = field(default_factory=lambda: [
        {"week": 8, "action": "price_cut", "intensity": 0.2},
        {"week": 20, "action": "promotion", "intensity": 0.3}
    ])

    # New product diffusion demand parameters (Bass model)
    diffusion_market_potential: float = 1000.0
    diffusion_innovation_coeff: float = 0.03  # Innovation coefficient p
    diffusion_imitation_coeff: float = 0.38   # Imitation coefficient q

    # Inventory-sensitive demand parameters
    inventory_sensitive_base_demand: float = 10.0
    inventory_sensitive_stockout_penalty: float = 0.3  # Stockout penalty coefficient
    inventory_sensitive_substitution_rate: float = 0.2  # Substitution rate


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"  # openai, anthropic, ollama, mock, deepseek, zhipu, etc.
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150
    timeout: int = 30
    max_retries: int = 3
    is_thinking_model: bool = False  # Whether model has built-in thinking/reasoning (R1, o1, etc.)

    # Fallback configurations
    fallback_configs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMProviderPreset:
    """Pre-configured LLM provider preset for common API services"""
    name: str                                    # Display name
    provider: str                                # Provider type
    model: str                                   # Model name
    base_url: Optional[str] = None               # API base URL
    api_key: Optional[str] = None                # API key (usually left empty for user to fill)
    description: str = ""                        # Description
    requires_api_key: bool = True                # Whether API key is required
    default_temperature: float = 0.7
    default_max_tokens: int = 150
    default_timeout: int = 30
    provider_label: str = ""                     # Short label for the provider
    is_thinking_model: bool = False              # Whether model has built-in thinking/reasoning (R1, o1, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "description": self.description,
            "requires_api_key": self.requires_api_key,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_timeout": self.default_timeout,
            "provider_label": self.provider_label,
            "is_thinking_model": self.is_thinking_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMProviderPreset":
        return cls(
            name=data.get("name", ""),
            provider=data.get("provider", "openai"),
            model=data.get("model", ""),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
            description=data.get("description", ""),
            requires_api_key=data.get("requires_api_key", True),
            default_temperature=data.get("default_temperature", 0.7),
            default_max_tokens=data.get("default_max_tokens", 150),
            default_timeout=data.get("default_timeout", 30),
            provider_label=data.get("provider_label", ""),
            is_thinking_model=data.get("is_thinking_model", False),
        )


# Built-in LLM provider presets for common API services
BUILTIN_LLM_PRESETS: Dict[str, LLMProviderPreset] = {
    # === OpenAI ===
    "openai_gpt4o": LLMProviderPreset(
        name="OpenAI GPT-4o",
        provider="openai",
        model="gpt-4o",
        description="OpenAI GPT-4o, latest multimodal flagship model",
        provider_label="OpenAI",
    ),
    "openai_gpt4o_mini": LLMProviderPreset(
        name="OpenAI GPT-4o-mini",
        provider="openai",
        model="gpt-4o-mini",
        description="OpenAI GPT-4o-mini, cost-effective small model",
        provider_label="OpenAI",
    ),
    "openai_gpt4_turbo": LLMProviderPreset(
        name="OpenAI GPT-4 Turbo",
        provider="openai",
        model="gpt-4-turbo",
        description="OpenAI GPT-4 Turbo with 128K context",
        provider_label="OpenAI",
    ),
    "openai_gpt35_turbo": LLMProviderPreset(
        name="OpenAI GPT-3.5 Turbo",
        provider="openai",
        model="gpt-3.5-turbo",
        description="OpenAI GPT-3.5 Turbo, fast and economical",
        provider_label="OpenAI",
    ),

    # === Anthropic ===
    "anthropic_opus": LLMProviderPreset(
        name="Claude Opus 4",
        provider="anthropic",
        model="claude-opus-4-20250514",
        description="Anthropic Claude Opus 4, most capable model",
        provider_label="Anthropic",
    ),
    "anthropic_sonnet": LLMProviderPreset(
        name="Claude Sonnet 4",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        description="Anthropic Claude Sonnet 4, balanced performance",
        provider_label="Anthropic",
    ),
    "anthropic_haiku": LLMProviderPreset(
        name="Claude Haiku 4",
        provider="anthropic",
        model="claude-haiku-4-20250501",
        description="Anthropic Claude Haiku 4, fast and lightweight",
        provider_label="Anthropic",
    ),

    # === DeepSeek ===
    "deepseek_chat": LLMProviderPreset(
        name="DeepSeek Chat (V3)",
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        description="DeepSeek V3 chat model, strong reasoning at low cost",
        provider_label="DeepSeek",
    ),
    "deepseek_reasoner": LLMProviderPreset(
        name="DeepSeek Reasoner (R1)",
        provider="deepseek",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        description="DeepSeek R1 reasoning model with chain-of-thought thinking",
        provider_label="DeepSeek",
        default_temperature=0.3,
        default_max_tokens=8000,
        default_timeout=120,
        is_thinking_model=True,
    ),

    # === Zhipu GLM ===
    "zhipu_glm4": LLMProviderPreset(
        name="Zhipu GLM-4",
        provider="zhipu",
        model="glm-4",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        description="Zhipu GLM-4 flagship model",
        provider_label="Zhipu",
    ),
    "zhipu_glm4_flash": LLMProviderPreset(
        name="Zhipu GLM-4 Flash",
        provider="zhipu",
        model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        description="Zhipu GLM-4 Flash, fast and free tier available",
        provider_label="Zhipu",
    ),

    # === Moonshot / Kimi ===
    "moonshot_kimi": LLMProviderPreset(
        name="Moonshot Kimi",
        provider="moonshot",
        model="moonshot-v1-8k",
        base_url="https://api.moonshot.cn/v1",
        description="Moonshot Kimi, long-context Chinese-optimized model",
        provider_label="Moonshot",
    ),

    # === Qwen (DashScope) ===
    "qwen_max": LLMProviderPreset(
        name="Qwen Max",
        provider="qwen",
        model="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Alibaba Qwen Max, top-tier Chinese LLM",
        provider_label="Qwen",
    ),
    "qwen_plus": LLMProviderPreset(
        name="Qwen Plus",
        provider="qwen",
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Alibaba Qwen Plus, balanced performance",
        provider_label="Qwen",
    ),

    # === OpenRouter ===
    "openrouter_auto": LLMProviderPreset(
        name="OpenRouter (Auto Route)",
        provider="openrouter",
        model="openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        description="OpenRouter with automatic model routing",
        provider_label="OpenRouter",
    ),

    # === Groq ===
    "groq_llama3": LLMProviderPreset(
        name="Groq Llama 3 70B",
        provider="groq",
        model="llama3-70b-8192",
        base_url="https://api.groq.com/openai/v1",
        description="Groq ultra-fast inference with Llama 3 70B",
        provider_label="Groq",
        default_temperature=0.3,
    ),

    # === Together AI ===
    "together_llama3": LLMProviderPreset(
        name="Together AI Llama 3 70B",
        provider="together",
        model="meta-llama/Llama-3-70b-chat-hf",
        base_url="https://api.together.xyz/v1",
        description="Together AI hosted Llama 3 70B",
        provider_label="Together AI",
    ),

    # === Generic OpenAI-Compatible ===
    "openai_compatible": LLMProviderPreset(
        name="Custom OpenAI-Compatible API",
        provider="openai_compatible",
        model="",
        base_url="http://localhost:8080/v1",
        description="Custom OpenAI-compatible API endpoint (e.g., vLLM, text-generation-webui, LocalAI, LiteLLM)",
        provider_label="Custom",
    ),

    # === Ollama (local) ===
    "ollama_gemma3": LLMProviderPreset(
        name="Ollama Gemma 3 27B",
        provider="ollama",
        model="gemma3:27b",
        base_url="http://localhost:11434",
        description="Google Gemma 3 27B running locally via Ollama",
        requires_api_key=False,
        default_temperature=0.3,
        provider_label="Ollama",
    ),
    "ollama_llama3": LLMProviderPreset(
        name="Ollama Llama 3 70B",
        provider="ollama",
        model="llama3:70b",
        base_url="http://localhost:11434",
        description="Meta Llama 3 70B running locally via Ollama",
        requires_api_key=False,
        default_temperature=0.3,
        provider_label="Ollama",
    ),
    "ollama_qwen2": LLMProviderPreset(
        name="Ollama Qwen 2 72B",
        provider="ollama",
        model="qwen2:72b",
        base_url="http://localhost:11434",
        description="Alibaba Qwen 2 72B running locally via Ollama",
        requires_api_key=False,
        default_temperature=0.3,
        provider_label="Ollama",
    ),

    # === GPT-OSS (local) ===
    "gptoss_20b": LLMProviderPreset(
        name="GPT-OSS 20B (Local)",
        provider="gpt_oss",
        model="gpt-oss:20b",
        base_url="http://localhost:11434",
        description="GPT-OSS 20B model running locally via Ollama",
        requires_api_key=False,
        default_temperature=0.3,
        default_max_tokens=50,
        provider_label="GPT-OSS",
    ),

    # === Mock (testing) ===
    "mock": LLMProviderPreset(
        name="Mock LLM (Testing)",
        provider="mock",
        model="mock-llm",
        description="Mock LLM for testing without API calls",
        requires_api_key=False,
        provider_label="Mock",
    ),
}


def get_preset_by_key(key: str) -> Optional[LLMProviderPreset]:
    """Get a built-in preset by its key"""
    return BUILTIN_LLM_PRESETS.get(key)


def get_presets_by_provider(provider: str) -> List[LLMProviderPreset]:
    """Get all presets for a specific provider"""
    return [p for p in BUILTIN_LLM_PRESETS.values() if p.provider == provider]


def get_all_preset_keys() -> List[str]:
    """Get all built-in preset keys"""
    return list(BUILTIN_LLM_PRESETS.keys())


def get_provider_options() -> List[str]:
    """Get all unique provider labels for dropdown menus"""
    seen = set()
    result = []
    for preset in BUILTIN_LLM_PRESETS.values():
        if preset.provider_label and preset.provider_label not in seen:
            seen.add(preset.provider_label)
            result.append(preset.provider_label)
    return result


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    total_weeks: int = 50
    lead_time: int = 2
    information_sharing: bool = False
    shared_demand_history: bool = False
    shared_inventory_levels: bool = False

    # Lead time configuration
    retailer_lead_time: int = 2
    wholesaler_lead_time: int = 2
    distributor_lead_time: int = 2
    manufacturer_lead_time: int = 2

    # Random seed
    random_seed: Optional[int] = None


# Import adaptive order limit configuration
from .adaptive_limits_config import AdaptiveLimitsConfig

@dataclass
class GameConfig:
    """Complete game configuration"""
    # Agent configurations
    retailer: AgentConfig = field(default_factory=AgentConfig)
    wholesaler: AgentConfig = field(default_factory=AgentConfig)
    distributor: AgentConfig = field(default_factory=AgentConfig)
    manufacturer: AgentConfig = field(default_factory=AgentConfig)

    # Demand configuration
    demand: DemandConfig = field(default_factory=DemandConfig)

    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Simulation configuration
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Adaptive order limit configuration
    adaptive_limits: AdaptiveLimitsConfig = field(default_factory=AdaptiveLimitsConfig)

    # Demand forecasting deduction mode (add-on, does not modify original prompts)
    enable_demand_forecasting: bool = False

    # Metadata
    name: str = "Default Config"
    description: str = "Standard beer game configuration"
    version: str = "1.0"


class ConfigManager:
    """Configuration manager"""

    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(exist_ok=True)

    def save_config(self, config: GameConfig, filename: str) -> None:
        """Save configuration to file"""
        config_path = self.config_dir / filename

        # Convert to dictionary
        config_dict = self._config_to_dict(config)

        # Choose format based on extension
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def load_config(self, filename: str) -> GameConfig:
        """Load configuration from file"""
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Choose format based on extension
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

        return self._dict_to_config(config_dict)

    def list_configs(self) -> List[str]:
        """List all config files"""
        return [f.name for f in self.config_dir.glob('*.json')] + \
               [f.name for f in self.config_dir.glob('*.yaml')] + \
               [f.name for f in self.config_dir.glob('*.yml')]

    def get_custom_presets_path(self) -> Path:
        """Get path to custom presets file"""
        return self.config_dir / "llm_custom_presets.json"

    def load_custom_presets(self) -> Dict[str, LLMProviderPreset]:
        """Load user's custom LLM provider presets"""
        path = self.get_custom_presets_path()
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                key: LLMProviderPreset.from_dict(val)
                for key, val in data.items()
            }
        except Exception:
            return {}

    def save_custom_presets(self, presets: Dict[str, LLMProviderPreset]) -> None:
        """Save user's custom LLM provider presets"""
        path = self.get_custom_presets_path()
        data = {key: preset.to_dict() for key, preset in presets.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_custom_preset(self, key: str, preset: LLMProviderPreset) -> None:
        """Add a custom preset"""
        presets = self.load_custom_presets()
        presets[key] = preset
        self.save_custom_presets(presets)

    def remove_custom_preset(self, key: str) -> bool:
        """Remove a custom preset. Returns True if removed."""
        presets = self.load_custom_presets()
        if key in presets:
            del presets[key]
            self.save_custom_presets(presets)
            return True
        return False

    def get_all_presets(self) -> Dict[str, LLMProviderPreset]:
        """Get all presets (built-in + custom)"""
        all_presets = dict(BUILTIN_LLM_PRESETS)
        all_presets.update(self.load_custom_presets())
        return all_presets

    def create_preset_configs(self) -> None:
        """Create preset configurations"""
        # Standard config
        standard_config = GameConfig(
            name="Standard Beer Game",
            description="Classic beer game configuration with step demand pattern"
        )
        self.save_config(standard_config, "standard.json")

        # Information sharing config
        info_sharing_config = GameConfig(
            name="Information Sharing Mode",
            description="Beer game with global information sharing enabled",
            simulation=SimulationConfig(
                information_sharing=True,
                shared_demand_history=True,
                shared_inventory_levels=True
            )
        )
        self.save_config(info_sharing_config, "info_sharing.json")

        # Rule-based agent config
        rule_based_config = GameConfig(
            name="Rule-Based Agent Mode",
            description="Beer game using rule-based agents",
            retailer=AgentConfig(agent_type=AgentType.RULE_BASED, base_stock_level=20),
            wholesaler=AgentConfig(agent_type=AgentType.RULE_BASED, base_stock_level=25),
            distributor=AgentConfig(agent_type=AgentType.RULE_BASED, base_stock_level=30),
            manufacturer=AgentConfig(agent_type=AgentType.RULE_BASED, base_stock_level=35)
        )
        self.save_config(rule_based_config, "rule_based.json")

        # Random demand config
        random_demand_config = GameConfig(
            name="Random Demand Mode",
            description="Beer game with random demand pattern",
            demand=DemandConfig(
                pattern_type=DemandPatternType.RANDOM,
                random_min=2,
                random_max=8,
                random_seed=42
            )
        )
        self.save_config(random_demand_config, "random_demand.json")
    
    def _config_to_dict(self, config: GameConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        def convert_dataclass(obj) -> Any:
            if hasattr(obj, '__dataclass_fields__'):
                result: Dict[str, Any] = {}
                for field_name, field_def in obj.__dataclass_fields__.items():
                    value = getattr(obj, field_name)
                    result[field_name] = convert_dataclass(value)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclass(v) for k, v in obj.items()}
            else:
                return obj
        
        return convert_dataclass(config)  # type: ignore
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> GameConfig:
        """Convert dictionary to config object"""
        # Handle enum types
        def convert_enums(data, target_class):
            if not hasattr(target_class, '__dataclass_fields__'):
                return data
            
            result = {}
            for field_name, field_def in target_class.__dataclass_fields__.items:
                if field_name not in data:
                    continue
                
                field_type = field_def.type
                value = data[field_name]
                
                # Handle enum
                if hasattr(field_type, '__bases__') and Enum in field_type.__bases__:
                    result[field_name] = field_type(value)
                # Handle nested dataclass
                elif hasattr(field_type, '__dataclass_fields__'):
                    result[field_name] = field_type(**convert_enums(value, field_type))
                else:
                    result[field_name] = value
            
            return result
        
        # Convert each component
        retailer_data = config_dict.get('retailer', {})
        if 'agent_type' in retailer_data:
            retailer_data['agent_type'] = AgentType(retailer_data['agent_type'])
        if 'cost_config' in retailer_data:
            retailer_data['cost_config'] = CostConfig(**retailer_data['cost_config'])
        
        wholesaler_data = config_dict.get('wholesaler', {})
        if 'agent_type' in wholesaler_data:
            wholesaler_data['agent_type'] = AgentType(wholesaler_data['agent_type'])
        if 'cost_config' in wholesaler_data:
            wholesaler_data['cost_config'] = CostConfig(**wholesaler_data['cost_config'])
        
        distributor_data = config_dict.get('distributor', {})
        if 'agent_type' in distributor_data:
            distributor_data['agent_type'] = AgentType(distributor_data['agent_type'])
        if 'cost_config' in distributor_data:
            distributor_data['cost_config'] = CostConfig(**distributor_data['cost_config'])
        
        manufacturer_data = config_dict.get('manufacturer', {})
        if 'agent_type' in manufacturer_data:
            manufacturer_data['agent_type'] = AgentType(manufacturer_data['agent_type'])
        if 'cost_config' in manufacturer_data:
            manufacturer_data['cost_config'] = CostConfig(**manufacturer_data['cost_config'])
        
        demand_data = config_dict.get('demand', {})
        if 'pattern_type' in demand_data:
            demand_data['pattern_type'] = DemandPatternType(demand_data['pattern_type'])
        
        # Handle adaptive order limit configuration
        adaptive_limits_data = config_dict.get('adaptive_limits', {})

        return GameConfig(
            retailer=AgentConfig(**retailer_data),
            wholesaler=AgentConfig(**wholesaler_data),
            distributor=AgentConfig(**distributor_data),
            manufacturer=AgentConfig(**manufacturer_data),
            demand=DemandConfig(**demand_data),
            llm=LLMConfig(**config_dict.get('llm', {})),
            simulation=SimulationConfig(**config_dict.get('simulation', {})),
            adaptive_limits=AdaptiveLimitsConfig.from_dict(adaptive_limits_data) if adaptive_limits_data else AdaptiveLimitsConfig(),
            name=config_dict.get('name', 'Default Config'),
            description=config_dict.get('description', ''),
            version=config_dict.get('version', '1.0')
        )


# Predefined configurations
def get_default_config() -> GameConfig:
    """Get default configuration"""
    return GameConfig()


def get_info_sharing_config() -> GameConfig:
    """Get information sharing configuration"""
    config = GameConfig()
    config.simulation.information_sharing = True
    config.simulation.shared_demand_history = True
    config.simulation.shared_inventory_levels = True
    config.name = "Information Sharing Mode"
    config.description = "Beer game with global information sharing enabled"
    return config


def get_rule_based_config() -> GameConfig:
    """Get rule-based agent configuration"""
    config = GameConfig()
    config.retailer.agent_type = AgentType.RULE_BASED
    config.retailer.base_stock_level = 20
    config.wholesaler.agent_type = AgentType.RULE_BASED
    config.wholesaler.base_stock_level = 25
    config.distributor.agent_type = AgentType.RULE_BASED
    config.distributor.base_stock_level = 30
    config.manufacturer.agent_type = AgentType.RULE_BASED
    config.manufacturer.base_stock_level = 35
    config.name = "Rule-Based Agent Mode"
    config.description = "Beer game using rule-based agents"
    return config