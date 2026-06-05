import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODELS = [
    {
        "name": "Groq (compound-mini)",
        "model": "groq/compound-mini",
        "provider": "groq",
        "temperature": 0.3,
        "max_tokens": 256,
        "timeout": 10,
        "enabled": True,
    },
    {
        "name": "Ollama (gemma4)",
        "model": "gemma4",
        "provider": "ollama",
        "temperature": 0.3,
        "max_tokens": 256,
        "timeout": 30,
        "enabled": True,
    },
    {
        "name": "Ollama (gemma4:e2b)",
        "model": "gemma4:e2b",
        "provider": "ollama",
        "temperature": 0.3,
        "max_tokens": 256,
        "timeout": 30,
        "enabled": True,
    },
    {
        "name": "Ollama (qwen3:1.7b)",
        "model": "qwen3:1.7b",
        "provider": "ollama",
        "temperature": 0.3,
        "max_tokens": 256,
        "timeout": 30,
        "enabled": True,
    },
]


def create_model(config: dict) -> Any:
    kwargs = {
        "model": config["model"],
        "model_provider": config["provider"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }
    if config["provider"] == "groq":
        kwargs["api_key"] = GROQ_API_KEY
    return init_chat_model(**kwargs)
