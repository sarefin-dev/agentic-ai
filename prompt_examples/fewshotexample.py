# Few-Shot Prompting Demo: Tone Rewriter

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common import MODELS, create_model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

TONE_EXAMPLES = [
    {"original": "Hey, can you send me that file?", "tone": "professional",
     "rewritten": "Could you please send me the file at your earliest convenience? Thank you."},
    {"original": "The meeting is at 3pm.", "tone": "enthusiastic",
     "rewritten": "Exciting news — our meeting is at 3pm! I can't wait to see everyone there! 🎉"},
    {"original": "I'm not sure if this is the right approach.", "tone": "confident",
     "rewritten": "While this approach has merit, I'd recommend we explore alternative strategies to ensure optimal results."},
    {"original": "We need to finish this by Friday.", "tone": "casual",
     "rewritten": "Hey team, let's try to wrap this up by Friday if possible! No rush though. 😊"},
    {"original": "That's a great idea!", "tone": "sarcastic",
     "rewritten": "Oh wow, what a completely original and never-before-thought-of idea. Truly groundbreaking stuff here."},
]

DYNAMIC_EXAMPLES = [
    {"original": "Can you send me the report?", "tone": "professional",
     "rewritten": "Could you please send me the report at your earliest convenience?"},
    {"original": "We need to talk.", "tone": "professional",
     "rewritten": "I would appreciate the opportunity to discuss this matter with you."},
    {"original": "Hello, how are you?", "tone": "casual", "rewritten": "Hey! What's up?"},
    {"original": "I will arrive shortly.", "tone": "casual", "rewritten": "Be there in a min!"},
    {"original": "Good job on the project.", "tone": "enthusiastic",
     "rewritten": "Wow, amazing work on the project! I'm so impressed! 🌟"},
    {"original": "The event is tomorrow.", "tone": "enthusiastic",
     "rewritten": "The event is TOMORROW! I'm so excited I can barely wait! 🎉"},
]

SYSTEM_PROMPT = (
    "You are a skilled writing assistant that rewrites text to match a specific tone. "
    "Maintain the original meaning while adapting the style. "
    "Respond with only the rewritten text, no explanations."
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def select_examples_by_tone(examples, tone, n=2):
    return [ex for ex in examples if ex["tone"] == tone][:n]


def _build_chain(examples, system_prompt, model_config):
    """Build a few-shot chain from examples, a system prompt, and a model config."""
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", 'Rewrite this in a {tone} tone: "{original}"'),
        ("ai", "{rewritten}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", 'Rewrite this in a {tone} tone: "{original}"'),
    ])
    return final_prompt | create_model(model_config) | StrOutputParser()


def select_model():
    """Prompt the user to choose a model from the MODELS registry."""
    enabled = [m for m in MODELS if m.get("enabled", True)]
    letters = "abcdefghijklmnopqrstuvwxyz"

    print("\n" + "=" * 60)
    print("SELECT A MODEL:")
    print("=" * 60)
    for i, m in enumerate(enabled):
        print(f"  [{letters[i]}] {m['name']}")
    print("=" * 60)

    valid = set(letters[: len(enabled)])
    while True:
        choice = input(f"Enter choice ({'/'.join(sorted(valid))}): ").strip().lower()
        if choice in valid:
            config = enabled[letters.index(choice)]
            print(f"Using: {config['name']}\n")
            return config
        print(f"Invalid choice. Enter one of: {', '.join(sorted(valid))}")


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #

def demo_tone_rewriter(model_config):
    """Rewrite several inputs through the full few-shot chain."""
    print("=" * 60)
    print(f"Few-Shot Tone Rewriter Demo ({model_config['name']})")
    print("=" * 60)

    chain = _build_chain(TONE_EXAMPLES, SYSTEM_PROMPT, model_config)

    cases = [
        {"original": "I think we should reconsider our strategy.", "tone": "professional"},
        {"original": "Your presentation was really good.", "tone": "enthusiastic"},
        {"original": "Sorry I'm late to the meeting.", "tone": "casual"},
        {"original": "This is the best solution we have.", "tone": "sarcastic"},
        {"original": "I don't think I can finish this on time.", "tone": "confident"},
    ]

    print("\nRewriting text with different tones...\n")
    print("-" * 60)

    for case in cases:
        print(f"\nOriginal:    {case['original']}")
        print(f"Target Tone: {case['tone'].upper()}")
        try:
            print(f"Rewritten:   {chain.invoke(case)}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 60)


def demo_compare_with_without_fewshot(model_config):
    """Show the output difference between zero-shot and few-shot for the same input."""
    print("\n" + "=" * 60)
    print(f"Comparison: Few-Shot vs Zero-Shot ({model_config['name']})")
    print("=" * 60)

    model = create_model(model_config)
    parser = StrOutputParser()

    zero_shot_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tone rewriter. Rewrite text to match the requested tone."),
        ("human", 'Rewrite this in a {tone} tone: "{original}"'),
    ])

    formal_examples = [
        {"original": "Can you help me?", "tone": "formal",
         "rewritten": "Would you be so kind as to assist me with this matter?"},
        {"original": "This is great!", "tone": "formal",
         "rewritten": "This is indeed most excellent and satisfactory."},
    ]

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", 'Rewrite this in a {tone} tone: "{original}"'),
        ("ai", "{rewritten}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=formal_examples,
    )
    few_shot_full_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tone rewriter. Rewrite text to match the requested tone."),
        few_shot_prompt,
        ("human", 'Rewrite this in a {tone} tone: "{original}"'),
    ])

    zero_shot_chain = zero_shot_prompt | model | parser
    few_shot_chain = few_shot_full_prompt | model | parser

    input_case = {"original": "I need this done by tomorrow.", "tone": "formal"}

    print(f'\nInput: "{input_case["original"]}"')
    print(f"Tone:  {input_case['tone'].upper()}\n")

    print("-" * 60)
    print("ZERO-SHOT (no examples):")
    try:
        print(f"→ {zero_shot_chain.invoke(input_case)}")
    except Exception as e:
        print(f"Error: {e}")

    print("-" * 60)
    print("FEW-SHOT (with examples):")
    try:
        print(f"→ {few_shot_chain.invoke(input_case)}")
    except Exception as e:
        print(f"Error: {e}")

    print("-" * 60)


def demo_dynamic_examples(model_config):
    """Select tone-matched examples dynamically before building each chain."""
    print("\n" + "=" * 60)
    print(f"Dynamic Example Selection Demo ({model_config['name']})")
    print("=" * 60)

    cases = [
        {"original": "Please review this document.", "tone": "professional"},
        {"original": "See you later.", "tone": "casual"},
        {"original": "We won the contract!", "tone": "enthusiastic"},
    ]

    print("\nSelecting relevant examples based on target tone...\n")

    for case in cases:
        selected = select_examples_by_tone(DYNAMIC_EXAMPLES, case["tone"])
        chain = _build_chain(selected, "You are a tone rewriter. Match the style of the examples.", model_config)

        print(f'Input: "{case["original"]}"')
        print(f"Tone:  {case['tone'].upper()} ({len(selected)} example(s) selected)")
        try:
            print(f"Result: {chain.invoke(case)}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 60)


def _run_all(model_config):
    for demo in [demo_tone_rewriter, demo_compare_with_without_fewshot, demo_dynamic_examples]:
        try:
            demo(model_config)
        except Exception as e:
            print(f"Error in {demo.__name__}: {e}")
    print("\nAll demos completed!")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("Few-Shot Prompting Examples")
    print("=" * 60)

    demos = {
        "1": ("Tone Rewriter Demo", demo_tone_rewriter),
        "2": ("Few-Shot vs Zero-Shot Comparison", demo_compare_with_without_fewshot),
        "3": ("Dynamic Example Selection", demo_dynamic_examples),
        "4": ("Run All Demos", _run_all),
    }

    while True:
        print("\n" + "=" * 60)
        print("SELECT A DEMO:")
        print("=" * 60)
        for key, (name, _) in demos.items():
            print(f"  [{key}] {name}")
        print("  [0] Exit")
        print("=" * 60)

        choice = input("\nEnter choice (0-4): ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice in demos:
            name, func = demos[choice]
            model_config = select_model()
            try:
                func(model_config)
            except Exception as e:
                print(f"Error in {name}: {e}")
        else:
            print("Invalid choice. Please enter 0-4.")


if __name__ == "__main__":
    main()
