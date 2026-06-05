# Person Brief: ask for a name, get a rich JSON profile using JsonOutputParser.
# Run from project root: python -m funny.person_brief

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from common import MODELS, create_model

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    "Return a JSON object about the person: {person}\n\n"
    "Include exactly these fields:\n"
    "  name         - full name\n"
    "  born         - birth year (integer, or null if unknown)\n"
    "  died         - death year (integer, or null if still alive / unknown)\n"
    "  nationality  - country or region\n"
    "  occupation   - primary role or title\n"
    "  known_for    - one-sentence tagline explaining why they are famous\n"
    "  notable_works - list of 3-5 notable works, discoveries, or achievements\n"
    "  fun_fact     - one surprising or little-known fact about them\n"
    "  famous_quote - their most recognisable quote, or null if none\n\n"
    "Respond with valid JSON only, no markdown fences."
)


def _print_profile(data: dict) -> None:
    name = data.get("name", "Unknown")
    born = data.get("born")
    died = data.get("died")
    nationality = data.get("nationality", "Unknown")
    occupation = data.get("occupation", "Unknown")
    known_for = data.get("known_for", "")
    notable_works = data.get("notable_works", [])
    fun_fact = data.get("fun_fact", "")
    famous_quote = data.get("famous_quote")

    lifespan = ""
    if born:
        lifespan = f" ({born}–{died if died else 'present'})"

    print(f"\n{'='*60}")
    print(f"  {name}{lifespan}")
    print(f"  {nationality}  |  {occupation}")
    print(f"{'='*60}")
    print(f"\n  {known_for}\n")

    if notable_works:
        print("  Notable works / achievements:")
        for item in notable_works:
            print(f"    * {item}")

    if fun_fact:
        print(f"\n  Fun fact: {fun_fact}")

    if famous_quote:
        print(f'\n  "{famous_quote}"')

    print()


def select_model() -> dict:
    enabled = [m for m in MODELS if m.get("enabled", True)]
    letters = "abcdefghijklmnopqrstuvwxyz"
    print("\nSelect a model:")
    for i, m in enumerate(enabled):
        print(f"  {letters[i]}. {m['name']}")
    raw = input("Choice: ").strip().lower()
    idx = letters.index(raw) if raw in letters[: len(enabled)] else 0
    chosen = enabled[idx]
    print(f"Using: {chosen['name']}\n")
    return chosen


def main() -> None:
    print("=" * 60)
    print("PERSON BRIEF  (powered by JsonOutputParser)")
    print("=" * 60)
    print("Ask about any real person and get a structured profile.\n")

    model_config = select_model()
    parser = JsonOutputParser()
    chain = PROMPT_TEMPLATE | create_model(model_config) | parser

    while True:
        person = input("Person name (or 'q' to quit) [Albert Einstein]: ").strip()
        if person.lower() == "q":
            print("Goodbye!")
            break
        if not person:
            person = "Albert Einstein"

        print(f"\nLooking up {person}...")
        try:
            result = chain.invoke({"person": person})
            _print_profile(result)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
