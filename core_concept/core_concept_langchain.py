# LangChain core concepts with environment-based model selection
# Supports: ollama, google_genai, openai, anthropic

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

print(f"Initializing {MODEL_NAME} from {MODEL_PROVIDER}...")
MODEL = init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER)


def _create_chain(prompt_template: str):
    """Create a basic chain from a prompt template string."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | MODEL | StrOutputParser()


def demo_basic_chain():
    """Single query processing demo."""
    chain = _create_chain(
        "You are a helpful assistant that answers questions about the world in one line.\n"
        "Question: {question}"
    )
    result = chain.invoke({"question": "What is the capital of Bangladesh?"})
    print(f"Result: {result}\n")


def demo_batch_chain():
    """Batch processing multiple queries."""
    chain = _create_chain(
        "You are a helpful assistant that answers questions about the world in one line.\n"
        "Question: {question}"
    )
    inputs = [
        {"question": "What is the capital of Bangladesh?"},
        {"question": "What is the largest planet in our solar system?"},
        {"question": "Who wrote Romeo and Juliet?"}
    ]
    results = chain.batch(inputs)
    for inp, result in zip(inputs, results):
        print(f"Input: {inp['question']} -> Result: {result}")
    print()


def demo_schema_chain():
    """Structured JSON output extraction."""
    class Person(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age as a number")
        city: str = Field(description="The city where the person lives")
        occupation: str = Field(description="The person's job or profession")

    parser = JsonOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Extract person information from the text below.\n"
        "{format_instructions}\n"
        "Text: {text}"
    )
    chain = prompt | MODEL | parser

    try:
        result = chain.invoke({
            "text": "John Smith is a 32-year-old software engineer living in San Francisco.",
            "format_instructions": parser.get_format_instructions()
        })
        if result is None:
            print("⚠️  Warning: Schema chain returned None\n")
            return

        print(f"Extracted Person Data: {result}\n")
        print(f"  Name: {result['name']}")
        print(f"  Age: {result['age']}")
        print(f"  City: {result['city']}")
        print(f"  Occupation: {result['occupation']}\n")

    except Exception as e:
        print(f"⚠️  Error parsing schema chain output: {type(e).__name__}: {e}\n")


def demo_gemma4_test():
    """Quick test with Gemma4 via Ollama."""
    try:
        gemma4 = init_chat_model(model="gemma4", model_provider="ollama")
        chain = _create_chain(
            "You are a helpful assistant. Answer this question concisely.\n"
            "Question: {question}",
            model=gemma4
        )
        print("Testing gemma4 model...\n")
        result = chain.invoke({"question": "What is the difference between AI and machine learning?"})
        print(f"Gemma4 Response: {result}\n")
    except Exception as e:
        print(f"⚠️  Error with gemma4 model: {type(e).__name__}: {e}\n")
        print("   Make sure gemma4 is pulled: ollama pull gemma4\n")


def demo_streaming_chain():
    """Real-time streaming output demo."""
    chain = _create_chain(
        "You are a helpful assistant. Provide a detailed answer to the question.\n"
        "Question: {question}"
    )
    print("Streaming response in real-time:\n")
    try:
        for chunk in chain.stream({"question": "What are the top 5 benefits of machine learning?"}):
            if chunk:
                print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"⚠️  Error during streaming: {type(e).__name__}: {e}\n")


def demo_marketing_tagline_generator():
    """Generate marketing taglines with custom model parameters."""
    prompt = ChatPromptTemplate.from_template(
        "You are a creative marketing expert. Generate 3 catchy and memorable taglines\n"
        "for the following product:\n\n"
        "Product Name: {product_name}\n"
        "Product Category: {category}\n"
        "Key Features: {features}\n"
        "Target Audience: {audience}\n\n"
        "Requirements:\n"
        "- Each tagline should be short (under 10 words)\n"
        "- Be creative, catchy, and memorable\n"
        "- Appeal to the target audience\n"
        "- Highlight unique value proposition\n\n"
        "Format your response as a numbered list (1. 2. 3.)"
    )

    tagline_model = init_chat_model(
        model=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        max_tokens=300,
        temperature=0.8,
        max_retries=3,
        timeout=30
    )

    chain = prompt | tagline_model | StrOutputParser()

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
        results = chain.batch(products)
        print("Generated Marketing Taglines:\n")
        for product, taglines in zip(products, results):
            print(f"📱 {product['product_name']} ({product['category']})")
            if taglines and taglines.strip():
                print(f"   {taglines}")
            else:
                print("   ⚠️  No taglines generated")
            print()
    except Exception as e:
        print(f"⚠️  Error generating marketing taglines: {type(e).__name__}: {e}\n")


def main():
    """Interactive demo menu."""
    print("=" * 60)
    print("LangChain Core Concepts")
    print("=" * 60)
    print(f"\n✓ Using: {MODEL_NAME} from {MODEL_PROVIDER}")
    print("=" * 60)

    demos = {
        "1": ("Basic Chain Demo", demo_basic_chain),
        "2": ("Batch Chain Demo", demo_batch_chain),
        "3": ("Schema Chain Demo", demo_schema_chain),
        "4": ("Streaming Chain Demo", demo_streaming_chain),
        "5": ("Marketing Tagline Generator", demo_marketing_tagline_generator),
        "6": ("Gemma4 Model Test", demo_gemma4_test),
        "7": ("Run All Demos", None),
    }

    while True:
        print("\n" + "=" * 60)
        print("SELECT A DEMO:")
        print("=" * 60)
        for key, (name, _) in demos.items():
            print(f"  [{key}] {name}")
        print("  [0] Exit")
        print("=" * 60)

        choice = input("\nEnter choice (0-7): ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "7":
            print("\n" + "=" * 60)
            print("Running all demos...")
            print("=" * 60)
            for key in ["1", "2", "3", "4", "5", "6"]:
                name, func = demos[key]
                print(f"\n[{key}] {name}:")
                print("-" * 60)
                try:
                    func()
                except Exception as e:
                    print(f"Error in {name}: {e}\n")
            print("\n" + "=" * 60)
            print("All demos completed!")
            print("=" * 60)
        elif choice in ["1", "2", "3", "4", "5", "6"]:
            name, func = demos[choice]
            print(f"\n[{choice}] {name}:")
            print("-" * 60)
            try:
                func()
            except Exception as e:
                print(f"Error: {e}\n")
        else:
            print("❌ Invalid choice. Please enter 0-7.")


if __name__ == "__main__":
    main()
7.")


if __name__ == "__main__":
    main()
