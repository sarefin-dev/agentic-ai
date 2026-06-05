# Model comparison demo: runs a translation task across multiple providers and models.
# Measures response time, handles retries, and prints a ranked summary table.
# Requires: pip install tenacity langchain-groq

import os
import time
from typing import Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model registry - easily extensible, all configuration in one place
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

# Test input for translation task
TEST_INPUT = {
    "input_language": "English",
    "output_language": "Bangla",
    "text": "I love my Bangladesh.",
}

# Translation prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "Translate the following text: {text}"),
    ]
)

output_parser = StrOutputParser()


def create_model(config: dict) -> Any:
    """Initialize a model using init_chat_model with configuration."""
    kwargs = {
        "model": config["model"],
        "model_provider": config["provider"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }
    if config["provider"] == "groq":
        kwargs["api_key"] = GROQ_API_KEY

    return init_chat_model(**kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_chain(chain: Any, inputs: dict) -> str:
    """Invoke chain with automatic retry on failure."""
    return chain.invoke(inputs)


def run_model_test(model_config: dict) -> dict:
    """Test a single model and return results."""
    model_name = model_config["name"]
    print(f"\nTesting: {model_name}")
    print("-" * 80)

    try:
        model = create_model(model_config)
        chain = prompt | model | output_parser

        start_time = time.time()
        response = invoke_chain(chain, TEST_INPUT)
        end_time = time.time()

        response_time = end_time - start_time

        print(f"Response: {response}")
        print(f"Time taken: {response_time:.2f} seconds")
        print(f"Temperature: {model_config['temperature']}")
        print("Status: ✓ Success")

        return {
            "response": response,
            "time": response_time,
            "status": "✓ Success",
            "attempts": 1,
        }

    except RetryError as e:
        print(f"Status: {model_name} Failed after 3 retry attempts")
        print(f"Last error: {str(e.last_attempt.exception)}")
        return {
            "response": None,
            "time": None,
            "status": "✗ Failed (3 retries)",
            "attempts": 3,
        }
    except Exception as e:
        print(f"Status: {model_name} Error")
        print(f"Error details: {str(e)}")
        return {
            "response": None,
            "time": None,
            "status": f"✗ Error: {type(e).__name__}",
            "attempts": 1,
        }


def print_summary(results: dict) -> None:
    """Print comparison summary table."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    successful = {k: v for k, v in results.items() if v["time"] is not None}

    if not successful:
        print("No successful model runs.")
        return

    sorted_results = sorted(successful.items(), key=lambda x: x[1]["time"])

    print(f"{'Model':<30} {'Time (s)':<12} {'Attempts':<10} {'Status':<20}")
    print("-" * 80)

    for model_name, data in sorted_results:
        print(
            f"{model_name:<30} {data['time']:<12.2f} {data['attempts']:<10} {data['status']:<20}"
        )

    fastest_time = sorted_results[0][1]["time"]
    print(f"\nRelative Speed (vs fastest: {sorted_results[0][0]}):")
    print("-" * 80)
    for model_name, data in sorted_results:
        multiplier = data["time"] / fastest_time
        status = "Fastest" if multiplier == 1.0 else f"{multiplier:.2f}x"
        print(f"  {model_name:<28} {status}")

    failed = {k: v for k, v in results.items() if v["time"] is None}
    if failed:
        print(f"\nFailed Models ({len(failed)}):")
        for model_name, data in failed.items():
            print(f"  - {model_name}: {data['status']}")


def main():
    """Run the model comparison."""
    print("=" * 80)
    print("MODEL COMPARISON WITH RETRY & CONFIGURATION")
    print("=" * 80)
    print(
        f"Task: Translate '{TEST_INPUT['text']}' from {TEST_INPUT['input_language']} to {TEST_INPUT['output_language']}"
    )
    print("=" * 80)

    enabled_models = [m for m in MODELS if m.get("enabled", True)]
    results = {model["name"]: run_model_test(model) for model in enabled_models}

    print_summary(results)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
