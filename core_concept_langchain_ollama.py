# Educational demonstration of LangChain core concepts using local Ollama models with init_chat_model.
# Shows LCEL (LangChain Expression Language) patterns for building AI chains.
# Prerequisites: Ollama server running on localhost:11434, llama3.2 model pulled.

"""
LANGCHAIN CORE CONCEPTS WITH OLLAMA - COMPREHENSIVE OVERVIEW
===========================================================

CONCEPT: LangChain Expression Language (LCEL) - Building Composable AI Chains
---------------------------------------------------------------------------

1. WHAT IS LCEL?
   - LCEL is a declarative way to compose LangChain components into processing pipelines
   - Uses the pipe operator (|) to chain components together
   - Creates a "runnable sequence" that processes input through multiple stages
   - Automatically handles async, batch processing, and streaming

2. CORE COMPONENTS IN A CHAIN:
   a) PromptTemplate: Takes user input and formats it into a structured prompt
   b) LLM (Language Model): Generates responses based on the prompt
   c) OutputParser: Transforms the LLM output into desired format (string, JSON, etc.)

3. CHAIN ARCHITECTURE: Input → Prompt Template → LLM → Output Parser → Result

4. CALLING SEQUENCE:
   Step 1: Create PromptTemplate with placeholders like {question}
   Step 2: Initialize LLM using init_chat_model with ollama provider
   Step 3: Initialize OutputParser to format the response
   Step 4: Compose chain using pipe operator: template | llm | parser
   Step 5: Invoke chain with dictionary containing the placeholder values

5. TECHNIQUES DEMONSTRATED:
   a) Single Query (invoke):
      - Process one input at a time
      - Use chain.invoke({"question": "..."})
      - Returns parsed output for that single input

   b) Batch Processing (batch):
      - Process multiple inputs at once
      - Use chain.batch([{"question": "..."}, {"question": "..."}])
      - Efficient for processing multiple queries in parallel
      - Better performance than looping with invoke()

6. WHY USE OLLAMA?
   - Runs locally without API costs
   - No internet dependency
   - Full control over the model
   - Great for development and testing
"""

from dotenv import load_dotenv

# Step 1: Import required LangChain components
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List

# Load environment variables (though Ollama doesn't need API keys)
load_dotenv()


def demo_basic_chain():
    """
    BASIC CHAIN DEMO - Single Query Processing
    ===========================================

    This demonstrates the simplest use case: asking one question and getting one answer.
    Flow: Question → Prompt Template → Ollama LLM → String Parser → Result
    """

    # Step 1: Define the prompt template using LCEL syntax
    # The {question} placeholder will be replaced with actual user input
    # This ensures consistent formatting for the LLM
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that answers questions about the world in one line.
        Question: {question}
        """
    )

    # Step 2: Initialize the Ollama LLM model using init_chat_model
    # init_chat_model provides a unified interface across different model providers
    # model_provider="ollama" specifies we're using Ollama
    # model="llama3.2" specifies which model to use
    # Make sure this model is pulled: ollama pull llama3.2
    model = init_chat_model(model="llama3.2", model_provider="ollama")

    # Step 3: Initialize the output parser
    # StrOutputParser converts the LLM's response into a plain string
    # This removes metadata and formatting, giving us just the text
    output_parser = StrOutputParser()

    # Step 4: Create the chain by composing components with pipe operator (|)
    # The pipe operator creates a "RunnableSequence" that flows data through each stage
    # Data flow: {question} → prompt_template → model → output_parser
    chain = prompt_template | model | output_parser

    # Step 5: Invoke the chain with input data
    # The dictionary key 'question' matches the placeholder in the prompt template
    result = chain.invoke({"question": "What is the capital of Bangladesh?"})

    # Step 6: Display the result
    print(f"Result: {result}")


def demo_batch_chain():
    """
    BATCH CHAIN DEMO - Multiple Query Processing
    =============================================

    This demonstrates efficient batch processing: asking multiple questions at once.
    The chain processes all inputs through the pipeline, which can be more efficient than
    looping and calling invoke() multiple times.

    Flow: Multiple Questions → Batch Processing → Multiple Results
    """

    # Step 1: Define the prompt template (same as basic chain)
    # This template will be used for each question in the batch
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that answers questions about the world in one line.
        Question: {question}
        """
    )

    # Step 2: Initialize the Ollama LLM model using init_chat_model
    # Same model configuration as the basic chain
    model = init_chat_model(model="llama3.2", model_provider="ollama")

    # Step 3: Initialize the output parser (same as basic chain)
    output_parser = StrOutputParser()

    # Step 4: Create the chain (same composition as basic chain)
    chain = prompt_template | model | output_parser

    # Step 5: Prepare batch inputs
    # Create a list of dictionaries, each containing input for one query
    # All dictionaries must have the same keys ('question' in this case)
    inputs = [
        {"question": "What is the capital of Bangladesh?"},
        {"question": "What is the largest planet in our solar system?"},
        {"question": "Who wrote Romeo and Juliet?"}
    ]

    # Step 6: Process batch using chain.batch()
    # batch() processes all inputs through the chain in sequence/parallel
    # Returns a list of results in the same order as inputs
    # More efficient than looping with invoke() for multiple queries
    results = chain.batch(inputs)

    # Step 7: Display results with their corresponding inputs
    # zip() pairs each input with its corresponding result
    for input_dict, result in zip(inputs, results):
        print(f"Input: {input_dict['question']} -> Result: {result}")


def demo_schema_chain():
    """
    SCHEMA CHAIN DEMO - Structured Output with JSON
    ================================================

    This demonstrates parsing LLM output into structured JSON format.
    Instead of getting raw text, we get validated, structured data that's
    easier to work with programmatically.

    Use case: Extracting information like name, age, city from text
    Flow: Question → Prompt → Ollama → JSON Parser → Structured Dictionary
    """

    # Step 1: Define a Pydantic model for structured output validation
    # This specifies exactly what fields we expect and their types
    # Pydantic validates that the LLM output matches this schema
    class Person(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age as a number")
        city: str = Field(description="The city where the person lives")
        occupation: str = Field(description="The person's job or profession")

    # Step 2: Create a JSON output parser using the Pydantic model
    # This parser converts raw LLM text into validated JSON/dict format
    output_parser = JsonOutputParser(pydantic_object=Person)

    # Step 3: Create the prompt with format instructions
    # get_format_instructions() automatically adds JSON schema instructions to the prompt
    # This tells the LLM exactly what JSON format to produce
    prompt_template = ChatPromptTemplate.from_template(
        """
        Extract person information from the text below.
        {format_instructions}
        Text: {text}
        """
    )

    # Step 4: Initialize Ollama model using init_chat_model
    model = init_chat_model(model="llama3.2", model_provider="ollama")

    # Step 5: Create the chain with schema-aware components
    # The chain now enforces structured output matching our Person model
    chain = prompt_template | model | output_parser

    # Step 6: Invoke with sample text
    # We pass the format instructions to the template
    result = chain.invoke({
        "text": "John Smith is a 32-year-old software engineer living in San Francisco.",
        "format_instructions": output_parser.get_format_instructions()
    })

    # Step 7: Display the structured result
    # Result is a dictionary with keys: name, age, city, occupation
    print("Extracted Person Data:")
    print(f"  Name: {result['name']}")
    print(f"  Age: {result['age']}")
    print(f"  City: {result['city']}")
    print(f"  Occupation: {result['occupation']}")


def demo_marketing_tag_line_generator():
    """
    MARKETING TAGLINE GENERATOR DEMO - Practical LLM Application
    ===========================================================

    This demonstrates a real-world use case: generating creative marketing
    taglines for products. Shows how chains can be used for content generation.

    Flow: Product Info → Prompt → Ollama → String Parser → Creative Taglines
    """

    # Step 1: Define the prompt template for tagline generation
    # This is specifically designed to get creative, marketing-focused output
    # The prompt sets context, constraints, and examples for better results
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

    # Step 2: Initialize the Ollama model using init_chat_model
    model = init_chat_model(model="llama3.2", model_provider="ollama")

    # Step 3: Use StrOutputParser to get clean text output
    # We want raw strings, not JSON, for this creative task
    output_parser = StrOutputParser()

    # Step 4: Create the chain for tagline generation
    chain = prompt_template | model | output_parser

    # Step 5: Prepare batch inputs for multiple products
    # Each product has its own context for tagline generation
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

    # Step 6: Process all products in batch
    # More efficient than looping with invoke() for multiple products
    tagline_results = chain.batch(products)

    # Step 7: Display results for each product
    # Pair each product with its generated taglines
    print("Generated Marketing Taglines:\n")
    for product, taglines in zip(products, tagline_results):
        print(f"📱 {product['product_name']} ({product['category']})")
        print(f"   {taglines}")
        print()


def main():
    """
    ENTRY POINT - Run the demonstrations
    ===================================
    """
    print("="*60)
    print("LangChain Core Concepts with Ollama")
    print("="*60)

    print("\n[1] Running Basic Chain Demo (Single Query):")
    print("-" * 60)
    demo_basic_chain()

    print("\n[2] Running Batch Chain Demo (Multiple Queries):")
    print("-" * 60)
    demo_batch_chain()

    print("\n[3] Running Schema Chain Demo (Structured JSON Output):")
    print("-" * 60)
    demo_schema_chain()

    print("\n[4] Running Marketing Tagline Generator (Practical Use Case):")
    print("-" * 60)
    demo_marketing_tag_line_generator()

    print("="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    # Ensure Ollama server is running before executing
    # Terminal command: ollama serve
    # And make sure the model is pulled: ollama pull llama3.2
    main()
