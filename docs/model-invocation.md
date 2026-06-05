# LangChain Model Invocation & Chaining

A reference for how models are initialized, called, chained, and hardened in this project.

---

## Prerequisites

```bash
pip install langchain langchain-core langchain-groq tenacity python-dotenv
# Ollama requires the Ollama desktop app running locally at http://localhost:11434
```

Environment variables (`.env`):

```
GROQ_API_KEY=gsk_...
```

---

## Vendor-Specific vs Generic Initialization

### Vendor-specific (explicit imports)

Each provider ships its own class. You import it directly and pass provider-specific kwargs.

```python
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

groq_model  = ChatGroq(api_key="...", model="compound-mini", temperature=0.3)
ollama_model = ChatOllama(model="gemma4", temperature=0.3)
```

**When to use:** prototyping with a single provider; you want IDE autocomplete on provider-specific options (e.g. `ChatGroq(reasoning_effort=...)`).

---

### Generic (`init_chat_model`)

`init_chat_model` resolves the provider at runtime from a string, so model config can live in data (a registry dict, environment variable, config file) rather than code.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="gemma4",
    model_provider="ollama",
    temperature=0.3,
    max_tokens=256,
)
```

**When to use:** model registries, multi-model scripts, user-selectable models. This is the pattern used in `model_comparison.py` via `create_model(config)`.

#### How `create_model` wraps it

```python
# model_comparison.py
def create_model(config: dict):
    kwargs = {
        "model":          config["model"],
        "model_provider": config["provider"],
        "temperature":    config["temperature"],
        "max_tokens":     config["max_tokens"],
    }
    if config["provider"] == "groq":
        kwargs["api_key"] = GROQ_API_KEY   # injected only where needed
    return init_chat_model(**kwargs)
```

The caller never needs to know which provider is active — it just holds a `model_config` dict and calls `create_model(model_config)`.

---

## Invocation Methods

All LangChain chat models share the same `Runnable` interface.

| Method | Description | Returns |
|---|---|---|
| `.invoke(input)` | Single synchronous call | `AIMessage` |
| `.stream(input)` | Token-by-token generator | `Iterator[AIMessageChunk]` |
| `.batch([input1, input2])` | Parallel calls | `List[AIMessage]` |
| `.ainvoke(input)` | Async single call | `Coroutine[AIMessage]` |

```python
# Direct invocation (returns AIMessage)
response = model.invoke("Translate: Hello")
print(response.content)

# Streaming
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

In practice you rarely call the model directly — you invoke a **chain** (see below), which routes input through a prompt template and output parser before reaching the model.

---

## Chaining with LCEL

LCEL (LangChain Expression Language) uses the `|` operator to compose `Runnable` objects left-to-right. Each stage's output becomes the next stage's input.

```
prompt_template | chat_model | output_parser
    dict              AIMessage      str
```

### Minimal chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tone rewriter."),
    ("human", 'Rewrite in a {tone} tone: "{text}"'),
])

chain = prompt | model | StrOutputParser()
result = chain.invoke({"tone": "formal", "text": "Can you help me?"})
```

### Few-shot chain (this project's pattern)

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

example_prompt = ChatPromptTemplate.from_messages([
    ("human", 'Rewrite this in a {tone} tone: "{original}"'),
    ("ai",    "{rewritten}"),
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,          # list of dicts
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    few_shot_prompt,            # injected between system and user turn
    ("human", 'Rewrite this in a {tone} tone: "{original}"'),
])

chain = final_prompt | create_model(model_config) | StrOutputParser()
```

`_build_chain` in `fewshotexample.py` encapsulates this so demos stay clean.

---

## Architectural Hardening

### Retry: Tenacity vs `with_retry`

#### Option A — Tenacity decorator (used in `model_comparison.py`)

Wraps the invoke call site. Good for isolating retry logic from chain construction.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def invoke_chain(chain, inputs):
    return chain.invoke(inputs)
```

| Attempt | Wait before retry |
|---|---|
| 1 → 2 | 2 s |
| 2 → 3 | 4 s |
| 3 → fail | `RetryError` raised |

`reraise=True` re-raises the original exception inside `RetryError.last_attempt.exception()`, which lets you log it specifically.

#### Option B — LangChain's `.with_retry()`

Attaches retry behaviour to any `Runnable` inline, keeping the chain expression self-contained.

```python
resilient_chain = (
    prompt
    | model.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
    | StrOutputParser()
)
```

**Trade-off summary:**

| | Tenacity | `.with_retry()` |
|---|---|---|
| Retry scope | Whole invoke call | Model node only |
| Custom callbacks | Full tenacity API | Limited to LangChain params |
| Error introspection | `RetryError.last_attempt` | Standard exception |
| Best for | Standalone scripts, external APIs | Inline chain composition |

---

### Rate Limiting

#### Tenacity as a rate-limit shield

Use `wait_exponential` or a fixed `wait_fixed` to back off when hitting provider rate limits (HTTP 429). This is implicit in the retry decorator above — a 429 triggers a retry with backoff.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=1, max=60),
)
def invoke_chain(chain, inputs):
    return chain.invoke(inputs)
```

#### `init_chat_model` rate limiter parameter

`init_chat_model` accepts a `rate_limiter` argument that accepts any object implementing `InMemoryRateLimiter`'s interface. This throttles calls **before** they go out, preventing 429s instead of recovering from them.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

limiter = InMemoryRateLimiter(
    requests_per_second=0.5,   # 1 request every 2 seconds
    check_every_n_seconds=0.1,
    max_bucket_size=2,
)

model = init_chat_model(
    model="groq/compound-mini",
    model_provider="groq",
    rate_limiter=limiter,
)
```

**Trade-off summary:**

| | Tenacity backoff | `InMemoryRateLimiter` |
|---|---|---|
| Approach | Reactive (retry after failure) | Proactive (throttle before sending) |
| Adds latency | Only on failure | Always (steady-state pacing) |
| 429 prevention | No (recovers from them) | Yes |
| Best for | Burst tolerance, flaky networks | Sustained throughput, quota-sensitive APIs |

For production use, combine both: the rate limiter prevents 429s under steady load; tenacity handles transient errors and unexpected bursts.

---

## Decision Tree

```
Need a model?
│
├─ Single provider, one-off script?
│   └─ Use vendor class (ChatGroq, ChatOllama)
│
└─ Multiple providers / user-selectable / config-driven?
    └─ Use init_chat_model via create_model(config)
        │
        ├─ Add few-shot examples?
        │   └─ FewShotChatMessagePromptTemplate → _build_chain(examples, prompt, config)
        │
        └─ Need resilience?
            ├─ Burst / flaky network → Tenacity @retry on invoke call
            ├─ Quota / rate limit → InMemoryRateLimiter on init_chat_model
            └─ Both → layer them (limiter on model, tenacity on invoke)
```
