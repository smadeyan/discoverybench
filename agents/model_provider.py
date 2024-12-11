from typing import Dict, Any
import requests
import json

class AnthropicModel(ModelInterface):
    """Implementation for Anthropic Claude"""
    def generate(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        return response.json()["content"][0]["text"]

class OpenAIModel(ModelInterface):
    """Implementation for OpenAI GPT models"""
    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        return response.json()["choices"][0]["message"]["content"]

def create_model(api_type: str, model_name: str, api_key: str, **kwargs) -> ModelInterface:
    """Factory function to create appropriate model interface"""
    if api_type == "anthropic":
        return AnthropicModel(model_name, api_key, **kwargs)
    elif api_type == "openai":
        return OpenAIModel(model_name, api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")