# src/healthcare_rag_llm/utils/api_config.py

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration data class"""
    api_key: str
    base_url: str
    provider: str

class APIConfigManager:
    """API configuration manager"""
    
    def __init__(self, config_path: str = "configs/api_config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"API configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_provider_config(self, provider_name: str) -> APIConfig:
        """Get configuration for a specific provider"""
        providers = self._config.get("api_providers", {})
        if provider_name not in providers:
            raise ValueError(f"Provider configuration not found: {provider_name}")
        
        provider_config = providers[provider_name]
        return APIConfig(
            api_key=provider_config["api_key"],
            base_url=provider_config["base_url"],
            provider=provider_config["provider"]
        )
    
    def get_model_config(self, model_name: str) -> APIConfig:
        """Get configuration for a specific model"""
        models = self._config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model configuration not found: {model_name}")
        
        model_config = models[model_name]
        provider_name = model_config["provider"]
        return self.get_provider_config(provider_name)
    
    def get_default_config(self) -> APIConfig:
        """Get default configuration"""
        default_provider = self._config.get("default_provider", "bltcy")
        return self.get_provider_config(default_provider)
    
    def list_available_providers(self) -> list:
        """List all available providers"""
        return list(self._config.get("api_providers", {}).keys())
    
    def list_available_models(self) -> list:
        """List all available models"""
        return list(self._config.get("models", {}).keys())


def load_api_config(config_path: str = "configs/api_config.yaml") -> Dict[str, Any]:
    """
    Convenience function to load API config as dictionary.

    Args:
        config_path: Path to API config YAML file

    Returns:
        Dictionary containing API configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"API configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)