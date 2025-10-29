"""
LLM Configuration Settings
Configures LLM models and settings for stock analysis reason generation
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM reason generation"""
    
    # API Settings
    api_key: str = None
    model: str = "llama-3.1-8b-instant"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    provider: str = "groq"  # groq or openai
    
    # Generation Settings
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: int = 30
    
    # Fallback Settings
    enable_fallback: bool = True
    min_response_length: int = 50
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        if not self.api_key:
            # Try Groq first, then OpenAI
            self.api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
        
        # Override with environment variables if present
        self.provider = os.getenv('LLM_PROVIDER', self.provider)
        self.model = os.getenv('LLM_MODEL', self.model)
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', self.max_tokens))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', self.temperature))
        
        # Set base URL based on provider
        if self.provider == "groq":
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        elif self.provider == "openai":
            self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Override with custom base URL if provided
        if os.getenv('LLM_BASE_URL'):
            self.base_url = os.getenv('LLM_BASE_URL')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for API calls"""
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return bool(self.api_key)
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create config from environment variables"""
        return cls()


# Default configuration
DEFAULT_LLM_CONFIG = LLMConfig.from_env()


# Model configurations for different use cases
MODEL_CONFIGS = {
    # Groq Models
    'llama-3.1-8b-instant': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.0002,
        'provider': 'groq'
    },
    'llama-3.1-70b-versatile': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.0007,
        'provider': 'groq'
    },
    'mixtral-8x7b-32768': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.0003,
        'provider': 'groq'
    },
    # OpenAI Models (fallback)
    'gpt-3.5-turbo': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.002,
        'provider': 'openai'
    },
    'gpt-4': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.03,
        'provider': 'openai'
    },
    'gpt-4-turbo': {
        'max_tokens': 500,
        'temperature': 0.7,
        'cost_per_1k_tokens': 0.01,
        'provider': 'openai'
    }
}


def get_llm_config(model: str = None, provider: str = None) -> LLMConfig:
    """Get LLM configuration for specified model and provider"""
    config = LLMConfig.from_env()
    
    # Override provider if specified
    if provider:
        config.provider = provider
        if provider == "groq":
            config.base_url = "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "openai":
            config.base_url = "https://api.openai.com/v1/chat/completions"
    
    if model and model in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[model]
        config.model = model
        config.max_tokens = model_config['max_tokens']
        config.temperature = model_config['temperature']
        
        # Set provider based on model if not already set
        if 'provider' in model_config and not provider:
            config.provider = model_config['provider']
            if config.provider == "groq":
                config.base_url = "https://api.groq.com/openai/v1/chat/completions"
            elif config.provider == "openai":
                config.base_url = "https://api.openai.com/v1/chat/completions"
    
    return config
