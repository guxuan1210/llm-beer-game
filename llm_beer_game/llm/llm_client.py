from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import json
import time

class LLMClient(ABC):
    """Abstract base class for LLM client"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.logger = logging.getLogger(f"LLMClient.{self.__class__.__name__}")

    @abstractmethod
    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150) -> str:
        """Generate text response"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        pass

class OpenAIClient(LLMClient):
    """OpenAI GPT client"""

    def __init__(self,
                 api_key: str,
                 model_name: str = "gpt-3.5-turbo",
                 base_url: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.base_url = base_url

        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self._available = True
        except ImportError:
            self.logger.error("OpenAI library not installed. Run: pip install openai")
            self._available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self._available = False

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150) -> str:
        """Generate response using OpenAI API with retry mechanism"""
        if not self._available:
            raise RuntimeError("OpenAI client not available")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[OpenAI] Attempting call {attempt + 1}/{max_retries}")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30  # Add timeout setting
                )

                # Log reasoning content if available (thinking models: DeepSeek R1, etc.)
                reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                if reasoning:
                    print(f"[OpenAI] Reasoning/thinking content ({len(reasoning)} chars): '{reasoning[:300]}...'")
                    # Save reasoning for debugging
                    if not hasattr(self, '_last_reasoning'):
                        self._last_reasoning = ""
                    self._last_reasoning = reasoning

                # Check response content
                content = response.choices[0].message.content
                print(f"[OpenAI] Raw response content: '{content}'")
                print(f"[OpenAI] Is content None: {content is None}")
                if content is None and reasoning:
                    print(f"[OpenAI] content is None but reasoning exists — thinking model may have exhausted token budget")

                if content is None:
                    print(f"[OpenAI] Response content is None, retrying {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait 1 second before retrying
                        continue
                    else:
                        return ""  # Last retry failed, return empty string

                result = content.strip()
                print(f"[OpenAI] Cleaned response: '{result}'")
                return result

            except Exception as e:
                self.logger.error(f"OpenAI API call failed (attempt {attempt + 1}): {e}")
                print(f"[OpenAI] API call exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retrying
                    continue
                else:
                    raise

    def is_available(self) -> bool:
        return self._available

class AnthropicClient(LLMClient):
    """Anthropic Claude client"""

    def __init__(self,
                 api_key: str,
                 model_name: str = "claude-3-haiku-20240307",
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self._thinking_budget = None  # Set to int to enable extended thinking

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self._available = True
        except ImportError:
            self.logger.error("Anthropic library not installed. Run: pip install anthropic")
            self._available = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            self._available = False

    def enable_thinking(self, budget_tokens: int = 1600):
        """Enable extended thinking for Claude models.

        Args:
            budget_tokens: Token budget for thinking (must be < max_tokens)
        """
        self._thinking_budget = budget_tokens

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150) -> str:
        """Generate response using Anthropic API"""
        if not self._available:
            raise RuntimeError("Anthropic client not available")

        try:
            # Check if thinking is enabled via kwargs
            thinking_config = None
            if hasattr(self, '_thinking_budget') and self._thinking_budget:
                thinking_config = {"type": "enabled", "budget_tokens": self._thinking_budget}

            kwargs = dict(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if thinking_config:
                kwargs["thinking"] = thinking_config

            response = self.client.messages.create(**kwargs)

            # Handle different content block types (thinking vs text)
            for block in response.content:
                if block.type == "text":
                    return block.text.strip()
            # Fallback: try first block if no text block found
            if response.content:
                return str(response.content[0]).strip()
            return ""

        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            raise

    def is_available(self) -> bool:
        return self._available

class OllamaClient(LLMClient):
    """Ollama local LLM client"""

    def __init__(self,
                 model_name: str = "gemma3:27b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.timeout = timeout  # Add timeout parameter

        try:
            import requests
            self.requests = requests
            # Test connection
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
        except ImportError:
            self.logger.error("Requests library not installed. Run: pip install requests")
            self._available = False
        except Exception as e:
            self.logger.warning(f"Ollama server not available: {e}")
            self._available = False

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150) -> str:
        """Generate response using Ollama API"""
        if not self._available:
            raise RuntimeError("Ollama client not available")

        try:
            prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout  # Use configured timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            raise

    def is_available(self) -> bool:
        return self._available

class GPTOSSClient(LLMClient):
    """GPT-OSS client, supports Ollama-deployed GPT-OSS models, uses Harmony format for output optimization"""

    def __init__(self,
                 model_name: str = "gpt-oss:20b",
                 base_url: str = "http://localhost:11434",
                 api_key: str = "ollama",
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key

        try:
            import requests
            self.session = requests.Session()
            # Test connection
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                self.logger.info(f"GPT-OSS client initialized successfully, model: {model_name}")
        except ImportError:
            self.logger.error("Requests library not installed. Run: pip install requests")
            self._available = False
        except Exception as e:
            self.logger.warning(f"GPT-OSS server not available: {e}")
            self._available = False

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.3,
                max_tokens: int = 50) -> str:
        """Generate response using GPT-OSS with Harmony format output optimization"""
        if not self._available:
            raise RuntimeError("GPT-OSS client is not available")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"GPT-OSS call (attempt {attempt + 1}/{max_retries})")
                self.logger.debug(f"Model: {self.model_name}")
                self.logger.debug(f"System prompt: {system_prompt[:100]}...")
                self.logger.debug(f"User prompt: {user_prompt[:100]}...")
                self.logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")

                # Use Harmony format API
                enhanced_system_prompt = f"{system_prompt}\n\nIMPORTANT: Output only a single number, no explanation, symbols, or other text."

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False
                }

                response = self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                self.logger.debug(f"API response structure: {result}")

                # Handle different response formats
                message_content = ""

                # Check if it's standard Ollama chat response format
                if isinstance(result, dict):
                    if 'message' in result:
                        # Standard chat API response
                        message_content = result.get('message', {}).get('content', '')
                    elif 'response' in result:
                        # Could be generate API response
                        message_content = result.get('response', '')
                    elif 'content' in result:
                        # Direct content response
                        message_content = result.get('content', '')
                else:
                    # If result is not a dict, it might be another format
                    message_content = str(result)

                if not message_content:
                    self.logger.warning(f"GPT-OSS returned empty content (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return ""

                self.logger.debug(f"Extracted message content: {message_content}")

                # Try to parse JSON format response content (Harmony format)
                try:
                    import json
                    if isinstance(message_content, str) and message_content.strip().startswith('{'):
                        response_data = json.loads(message_content)

                        # Extract response.content field
                        final_content = response_data.get('response', {}).get('content', '')
                        if final_content:
                            # Extract number from final content
                            import re
                            numbers = re.findall(r'\d+', final_content)
                            if numbers:
                                self.logger.info(f"GPT-OSS Harmony format successfully extracted number: {numbers[0]}")
                                return numbers[0]
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    # If not JSON format, extract numbers directly from raw content
                    self.logger.debug(f"JSON parsing failed, using direct extraction: {e}")

                # Extract numbers directly from message content (fallback)
                import re
                numbers = re.findall(r'\d+', str(message_content))
                if numbers:
                    self.logger.info(f"GPT-OSS directly extracted number: {numbers[0]}, raw response: {str(message_content)[:100]}...")
                    return numbers[0]

                self.logger.warning(f"GPT-OSS did not return valid number, raw response: {message_content}")
                return ""

            except Exception as e:
                self.logger.error(f"GPT-OSS Harmony API call failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return ""

    def is_available(self) -> bool:
        """Check if GPT-OSS service is available"""
        if not self._available:
            return False

        try:
            # Use Ollama native API to check availability
            payload = {
                "model": self.model_name,
                "prompt": "test",
                "stream": False,
                "options": {
                    "num_predict": 1
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.debug(f"GPT-OSS availability check failed: {e}")
            return False

class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""

    def __init__(self, model_name: str = "mock-llm", **kwargs):
        super().__init__(model_name, **kwargs)
        self._available = True
        self._response = None  # Used for setting a fixed response

    def set_response(self, response: str):
        """Set fixed response content"""
        self._response = response

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150) -> str:
        """Generate mock response"""
        # If fixed response is set, return it directly
        if self._response is not None:
            return self._response

        # Simple heuristic rules to generate response
        import re

        # Extract numeric information from user prompt
        inventory_match = re.search(r'(?:Current inventory|Inventory):\s*(\d+)', user_prompt)
        demand_match = re.search(r'(?:Current period demand|Demand):\s*(\d+)', user_prompt)
        backorder_match = re.search(r'(?:Backorders|Backorder):\s*(\d+)', user_prompt)

        inventory = int(inventory_match.group(1)) if inventory_match else 10
        demand = int(demand_match.group(1)) if demand_match else 5
        backorder = int(backorder_match.group(1)) if backorder_match else 0

        # Simple decision logic
        if backorder > 0:
            order = demand + backorder + 2  # Replenish backorders and add buffer
        elif inventory < demand:
            order = demand * 2  # Order more when inventory is insufficient
        elif inventory > demand * 3:
            order = max(1, demand // 2)  # Reduce ordering when inventory is excessive
        else:
            order = demand  # Under normal circumstances, order equals demand

        return str(order)

    def is_available(self) -> bool:
        return True

class LLMClientFactory:
    """LLM client factory"""

    # Supported provider aliases
    PROVIDER_ALIASES = {
        "deepseek": "openai",       # DeepSeek uses OpenAI-compatible API
        "zhipu": "openai",          # Zhipu GLM uses OpenAI-compatible API
        "moonshot": "openai",       # Moonshot/Kimi uses OpenAI-compatible API
        "qwen": "openai",           # Qwen (DashScope) uses OpenAI-compatible API
        "openai_compatible": "openai",  # Generic OpenAI-compatible API
        "openrouter": "openai",     # OpenRouter uses OpenAI-compatible API
        "groq": "openai",           # Groq uses OpenAI-compatible API
        "together": "openai",       # Together AI uses OpenAI-compatible API
    }

    @staticmethod
    def create_client(provider: str, **kwargs) -> LLMClient:
        """Create LLM client

        Args:
            provider: Provider ('openai', 'anthropic', 'ollama', 'gpt_oss', 'mock',
                       'deepseek', 'zhipu', 'moonshot', 'qwen', 'openai_compatible',
                       'openrouter', 'groq', 'together')
            **kwargs: Client parameters

        Returns:
            LLM client instance
        """
        provider_lower = provider.lower()

        # Check provider aliases (OpenAI-compatible providers)
        if provider_lower in LLMClientFactory.PROVIDER_ALIASES:
            mapped = LLMClientFactory.PROVIDER_ALIASES[provider_lower]
            if mapped == "openai":
                return OpenAIClient(**kwargs)

        if provider_lower == "openai":
            return OpenAIClient(**kwargs)
        elif provider_lower == "anthropic":
            return AnthropicClient(**kwargs)
        elif provider_lower == "ollama":
            return OllamaClient(**kwargs)
        elif provider_lower == "gpt_oss" or provider_lower == "gpt-oss":
            return GPTOSSClient(**kwargs)
        elif provider_lower == "mock":
            return MockLLMClient(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LLMClient:
        """Create LLM client from config

        Args:
            config: Config dictionary containing provider and other parameters

        Returns:
            LLM client instance
        """
        provider = config.pop("provider")
        return LLMClientFactory.create_client(provider, **config)

class LLMManager:
    """LLM manager, supporting multiple clients and failover"""

    def __init__(self, clients: List[LLMClient]):
        self.clients = clients
        self.current_client_index = 0
        self.logger = logging.getLogger("LLMManager")

    def get_available_client(self) -> Optional[LLMClient]:
        """Get an available LLM client"""
        for i, client in enumerate(self.clients):
            if client.is_available():
                self.current_client_index = i
                return client
        return None

    def generate(self,
                system_prompt: str,
                user_prompt: str,
                temperature: float = 0.1,
                max_tokens: int = 150,
                retry_count: int = 2) -> str:
        """Generate response with failover support"""
        last_error = None

        for attempt in range(retry_count + 1):
            client = self.get_available_client()
            if not client:
                raise RuntimeError("No available LLM clients")

            try:
                response = client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

                # Try next client
                if attempt < retry_count:
                    self.current_client_index = (self.current_client_index + 1) % len(self.clients)
                    time.sleep(1)  # Brief delay

        raise RuntimeError(f"All LLM clients failed. Last error: {last_error}")

    def is_available(self) -> bool:
        """Check if any client is available"""
        return any(client.is_available() for client in self.clients)
