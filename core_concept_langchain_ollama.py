# LangChain core concepts using local Ollama models
# Prerequisites: Ollama server running, llama3.2 model pulled

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()


def _create_chain(prompt_template: str, model_name: str = "llama3.2"):
    """Create a basic chain with Ollama model."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = init_chat_model(model=model_name, model_provider="ollama")
    return prompt | model | StrOutputParser()


def demo_basic_chain():
    """Single query processing demo."""
    chain = _create_chain(
        "You are a helpful assistant that answers questions about the world in one line.\n"
        "Question: {question}"
    )
    result = chain.invoke({"question": "What is the capital of Bangladesh?"})
    print(f"Result: {result}")


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
    model = init_chat_model(model="llama3.2", model_provider="ollama")
    chain = prompt | model | parser

    result = chain.invoke({
        "text": "John Smith is a 32-year-old software engineer living in San Francisco.",
        "format_instructions": parser.get_format_instructions()
    })

    print("Extracted Person Data:")
    print(f"  Name: {result['name']}")
    print(f"  Age: {result['age']}")
    print(f"  City: {result['city']}")
    print(f"  Occupation: {result['occupation']}")


def demo_marketing_tagline_generator():
    """Generate marketing taglines for products."""
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

    model = init_chat_model(model="llama3.2", model_provider="ollama")
    chain = prompt | model | StrOutputParser()

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

    results = chain.batch(products)

    print("Generated Marketing Taglines:\n")
    for product, taglines in zip(products, results):
        print(f"📱 {product['product_name']} ({product['category']})")
        print(f"   {taglines}")
        print()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("LangChain Core Concepts with Ollama")
    print("=" * 60)

    demos = [
        ("Basic Chain Demo", demo_basic_chain),
        ("Batch Chain Demo", demo_batch_chain),
        ("Schema Chain Demo", demo_schema_chain),
        ("Marketing Tagline Generator", demo_marketing_tagline_generator),
    ]

    for name, func in demos:
        print(f"\n[ ] {name}:")
        print("-" * 60)
        func()

    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
