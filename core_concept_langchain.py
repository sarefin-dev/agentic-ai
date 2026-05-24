# Educational demonstration of LangChain core concepts with environment-based model selection.
# Shows LCEL (LangChain Expression Language) patterns with configurable model providers.
# Supports multiple model providers: ollama, google_genai, openai, anthropic, etc.

"""
LANGCHAIN CORE CONCEPTS WITH ENVIRONMENT-BASED MODEL SELECTION
==============================================================

CONCEPT: Flexible Model Provider Configuration
----------------------------------------------

This module demonstrates LangChain chains with environment-based model selection.
Instead of hardcoding a specific model or provider, we read from .env to choose
which model and provider to use at runtime.

SUPPORTED PROVIDERS:
  - "ollama": Local models via Ollama (no API key needed)
  - "google_genai": Google's Gemini models (requires GOOGLE_API_KEY)
  - "openai": OpenAI models (requires OPENAI_API_KEY)
  - "anthropic": Anthropic's Claude (requires ANTHROPIC_API_KEY)

ENVIRONMENT VARIABLES:
  - MODEL_PROVIDER: Which provider to use (default: "ollama")
  - MODEL_NAME: Model identifier (e.g., "llama3.2", "gemini-2.0-flash")
  - GOOGLE_API_KEY: API key for Google Gemini (if using google_genai)
  - OPENAI_API_KEY: API key for OpenAI (if using openai)
  - ANTHROPIC_API_KEY: API key for Claude (if using anthropic)
"""

import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


# Initialize model once at module level
model_provider = os.getenv("MODEL_PROVIDER", "ollama")
model_name = os.getenv("MODEL_NAME", "llama3.2")

print(f"Initializing {model_name} from {model_provider}...")
model = init_chat_model(
    model=model_name,
    model_provider=model_provider
)


def demo_basic_chain():
    """
    BASIC CHAIN DEMO - Single Query Processing with Configurable Model
    ===================================================================

    Shows how to use environment-configured model in a simple chain.
    Flow: Question → Prompt Template → Model → String Parser → Result
    """

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that answers questions about the world in one line.
        Question: {question}
        """
    )

    # Using module-level initialized model
    output_parser = StrOutputParser()

    chain = prompt_template | model | output_parser

    result = chain.invoke({"question": "What is the capital of Bangladesh?"})

    print(f"Result: {result}\n")


def demo_batch_chain():
    """
    BATCH CHAIN DEMO - Multiple Query Processing with Configurable Model
    ====================================================================

    Demonstrates efficient batch processing with environment-configured model.
    Flow: Multiple Questions → Batch Processing → Multiple Results
    """

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that answers questions about the world in one line.
        Question: {question}
        """
    )

    # Using module-level initialized model
    output_parser = StrOutputParser()

    chain = prompt_template | model | output_parser

    inputs = [
        {"question": "What is the capital of Bangladesh?"},
        {"question": "What is the largest planet in our solar system?"},
        {"question": "Who wrote Romeo and Juliet?"}
    ]

    results = chain.batch(inputs)

    for input_dict, result in zip(inputs, results):
        print(f"Input: {input_dict['question']} -> Result: {result}")
    print()


def demo_schema_chain():
    """
    SCHEMA CHAIN DEMO - Structured Output with JSON
    ===============================================

    Demonstrates structured output with environment-configured model.
    Flow: Text → Prompt → Model → JSON Parser → Structured Dictionary
    """

    class Person(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age as a number")
        city: str = Field(description="The city where the person lives")
        occupation: str = Field(description="The person's job or profession")

    output_parser = JsonOutputParser(pydantic_object=Person)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Extract person information from the text below.
        {format_instructions}
        Text: {text}
        """
    )

    # Using module-level initialized model

    chain = prompt_template | model | output_parser

    try:
        result = chain.invoke({
            "text": "John Smith is a 32-year-old software engineer living in San Francisco.",
            "format_instructions": output_parser.get_format_instructions()
        })

        if result is None:
            print("⚠️  Warning: Schema chain returned None. This may indicate:")
            print("   - Model response was not valid JSON")
            print("   - Ollama server may be having issues")
            print("   - Try restarting Ollama or using a different provider\n")
            return

        print(f"Extracted Person Data: {result}\n")
        print(f"  Name: {result['name']}")
        print(f"  Age: {result['age']}")
        print(f"  City: {result['city']}")
        print(f"  Occupation: {result['occupation']}\n")

    except Exception as e:
        print(f"⚠️  Error parsing schema chain output: {type(e).__name__}: {e}\n")
        print("   This typically means the model response format didn't match expected JSON.\n")


def demo_streaming_chain():
    """
    STREAMING CHAIN DEMO - Real-time Output Streaming
    ================================================

    Demonstrates streaming output from the model in real-time.
    Instead of waiting for the full response, chunks arrive as the model generates them.
    Useful for creating responsive UIs and showing progress for long-running operations.

    Flow: Question → Prompt → Model → String Parser → Streamed Chunks
    """

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant. Provide a detailed answer to the question.
        Question: {question}
        """
    )

    output_parser = StrOutputParser()
    chain = prompt_template | model | output_parser

    print("Streaming response in real-time:\n")

    try:
        full_response = ""
        for chunk in chain.stream({"question": "What are the top 5 benefits of machine learning?"}):
            if chunk:
                print(chunk, end="", flush=True)
                full_response += chunk

        print("\n")

    except Exception as e:
        print(f"⚠️  Error during streaming: {type(e).__name__}: {e}\n")


def demo_marketing_tagline_generator():
    """
    MARKETING TAGLINE GENERATOR - Practical Use Case with Custom Parameters
    ======================================================================

    Real-world use case: generating marketing taglines with configurable model.
    Demonstrates custom parameters: max_tokens, temperature, max_retries, timeout.
    Flow: Product Info → Prompt → Model → String Parser → Creative Taglines
    """

    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a creative marketing expert. Generate 3 catchy and memorable taglines
        for the following product:

        Product Name: {product_name}
        Product Category: {category}
        Key Features: {features}
        Target Audience: {audience}

        Requirements:
        - Each tagline should be short (under 10 words)
        - Be creative, catchy, and memorable
        - Appeal to the target audience
        - Highlight unique value proposition

        Format your response as a numbered list (1. 2. 3.)
        """
    )

    # Initialize model with custom parameters for creative tasks
    tagline_model = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        max_tokens=300,           # Limit output length for concise taglines
        temperature=0.8,          # Higher temperature for more creative output
        max_retries=3,            # Retry up to 3 times on failure
        timeout=30                # 30 second timeout per request
    )

    output_parser = StrOutputParser()

    # Use tagline_model with custom parameters for this chain
    chain = prompt_template | tagline_model | output_parser

    products = [
        {
            "product_name": "EcoBottle",
            "category": "Sustainable Water Bottles",
            "features": "Biodegradable, keeps water cold 24 hours, leak-proof",
            "audience": "Eco-conscious millennials and Gen Z"
        },
        {
            "product_name": "FitTrack Pro",
            "category": "Fitness Wearable",
            "features": "Heart rate monitoring, sleep tracking, AI coaching",
            "audience": "Fitness enthusiasts and health-conscious professionals"
        },
        {
            "product_name": "CloudPillow",
            "category": "Smart Bedding",
            "features": "Temperature control, white noise, memory foam",
            "audience": "Insomniacs and quality sleep seekers"
        }
    ]

    try:
        tagline_results = chain.batch(products)

        print("Generated Marketing Taglines:\n")
        for product, taglines in zip(products, tagline_results):
            if taglines is None or not taglines.strip():
                print(f"📱 {product['product_name']} ({product['category']})")
                print("   ⚠️  No taglines generated (model may be overloaded or unresponsive)")
            else:
                print(f"📱 {product['product_name']} ({product['category']})")
                print(f"   {taglines}")
            print()

    except Exception as e:
        print(f"⚠️  Error generating marketing taglines: {type(e).__name__}: {e}\n")


def main():
    """
    ENTRY POINT - Interactive Demo Menu
    ===================================
    """
    print("="*60)
    print("LangChain Core Concepts with Environment-Based Model Selection")
    print("="*60)
    print(f"\n✓ Using: {model_name} from {model_provider}")
    print("="*60)

    demos = {
        "1": ("Basic Chain Demo (Single Query)", demo_basic_chain),
        "2": ("Batch Chain Demo (Multiple Queries)", demo_batch_chain),
        "3": ("Schema Chain Demo (Structured JSON Output)", demo_schema_chain),
        "4": ("Streaming Chain Demo (Real-time Output)", demo_streaming_chain),
        "5": ("Marketing Tagline Generator (Practical Use Case)", demo_marketing_tagline_generator),
        "6": ("Run All Demos", None),
    }

    while True:
        print("\n" + "="*60)
        print("SELECT A DEMO TO RUN:")
        print("="*60)
        for key, (name, _) in demos.items():
            print(f"  [{key}] {name}")
        print("  [0] Exit")
        print("="*60)

        choice = input("\nEnter your choice (0-6): ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break

        elif choice == "6":
            print("\n" + "="*60)
            print("Running all demos...")
            print("="*60)
            for key in ["1", "2", "3", "4", "5"]:
                name, func = demos[key]
                print(f"\n[{key}] {name}:")
                print("-" * 60)
                try:
                    func()
                except Exception as e:
                    print(f"Error in {name}: {e}\n")
            print("\n" + "="*60)
            print("All demos completed!")
            print("="*60)

        elif choice in ["1", "2", "3", "4", "5"]:
            name, func = demos[choice]
            print(f"\n[{choice}] {name}:")
            print("-" * 60)
            try:
                func()
            except Exception as e:
                print(f"Error: {e}\n")

        else:
            print("❌ Invalid choice. Please enter 0-6.")


if __name__ == "__main__":
    main()
