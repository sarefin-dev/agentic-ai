# Output Parser Demo: four LangChain parser strategies side by side.
# Run from project root: python -m outputformatter.output_parsers_demo

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from common import MODELS, create_model


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's primary occupation")


class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #

def demo_str_parser(model_config: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Demo 1 — StrOutputParser  ({model_config['name']})")
    print("="*60)
    print("Chain: prompt | llm | StrOutputParser()\n"
          "Returns the model response as a plain Python str.\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise technical advisor."),
        ("human", "What makes a great software engineer? Give exactly 3 bullet points."),
    ])
    parser = StrOutputParser()
    chain = prompt | create_model(model_config) | parser

    result = chain.invoke({})
    print(f"Type  : {type(result).__name__}")
    print(f"Output:\n{result}")


def demo_json_parser(model_config: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Demo 2 — JsonOutputParser  ({model_config['name']})")
    print("="*60)
    print("Chain: prompt | llm | JsonOutputParser()\n"
          "Parses the model response into a Python dict (no schema enforced).\n")

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
    )
    parser = JsonOutputParser()
    chain = prompt | create_model(model_config) | parser

    result = chain.invoke({"description": "Marie Curie"})
    print(f"Type  : {type(result).__name__}")
    print(f"Output: {result}")
    print(f"  name={result.get('name')}, age={result.get('age')}, "
          f"occupation={result.get('occupation')}")


def demo_pydantic_parser(model_config: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Demo 3 — PydanticOutputParser  ({model_config['name']})")
    print("="*60)
    print("Chain: prompt | llm | PydanticOutputParser(pydantic_object=Person)\n"
          "Injects format instructions via .partial(), validates into a Person object.\n")

    parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with 'name', 'age', and 'occupation' for: {description}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | create_model(model_config) | parser

    result = chain.invoke({"description": "Albert Einstein"})
    print(f"Type  : {type(result).__name__}")
    print(f"Output: {result}")
    print(f"  name={result.name}, age={result.age}, occupation={result.occupation}")


def demo_structured_output(model_config: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Demo 4 — with_structured_output  ({model_config['name']})")
    print("="*60)
    print("structured_model = llm.with_structured_output(MovieReview)\n"
          "No parser in the chain — schema is bound to the model itself.\n")

    llm = create_model(model_config)
    structured_model = llm.with_structured_output(MovieReview)

    result = structured_model.invoke("Give me a brief review of the movie Interstellar.")
    print(f"Type  : {type(result).__name__}")
    print(f"Output: {result}")
    print(f"  title={result.title}, rating={result.rating}/10")
    print(f"  review={result.review}")


def _run_all(model_config: dict) -> None:
    demo_str_parser(model_config)
    demo_json_parser(model_config)
    demo_pydantic_parser(model_config)
    demo_structured_output(model_config)


# --------------------------------------------------------------------------- #
# Model selector & main loop
# --------------------------------------------------------------------------- #

DEMOS = [
    ("StrOutputParser — plain text", demo_str_parser),
    ("JsonOutputParser — dict", demo_json_parser),
    ("PydanticOutputParser — Person schema", demo_pydantic_parser),
    ("with_structured_output — MovieReview schema", demo_structured_output),
    ("Run all demos", _run_all),
]


def select_model() -> dict:
    enabled = [m for m in MODELS if m.get("enabled", True)]
    letters = "abcdefghijklmnopqrstuvwxyz"
    print("\nSelect a model:")
    for i, m in enumerate(enabled):
        print(f"  {letters[i]}. {m['name']}")
    raw = input("Choice: ").strip().lower()
    idx = letters.index(raw) if raw in letters[: len(enabled)] else 0
    chosen = enabled[idx]
    print(f"Using: {chosen['name']}")
    return chosen


def main() -> None:
    print("=" * 60)
    print("OUTPUT PARSER DEMO")
    print("=" * 60)

    while True:
        print("\nDemos:")
        for i, (label, _) in enumerate(DEMOS, 1):
            print(f"  {i}. {label}")
        print("  q. Quit")

        choice = input("\nSelect demo: ").strip().lower()
        if choice == "q":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(DEMOS)):
            print("Invalid choice.")
            continue

        model_config = select_model()
        _, fn = DEMOS[int(choice) - 1]
        try:
            fn(model_config)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
