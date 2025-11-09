# LangChain 1.0 Model Provider Configuration

This guide covers configuration for various LLM providers in LangChain 1.0.

## Environment Setup

### Required Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here  # Optional

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI
GOOGLE_API_KEY=your_google_api_key_here

# Hugging Face
HUGGINGFACEHUB_API_KEY=your_huggingface_api_key_here

# Cohere
COHERE_API_KEY=your_cohere_api_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# AWS Bedrock
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

## Provider Configuration

### 1. OpenAI

**Installation:**
```bash
pip install langchain-openai
```

**Basic Usage:**
```python
from langchain_openai import ChatOpenAI, OpenAI

# Chat model (recommended)
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
    openai_api_key="your-api-key",  # Optional if in env
    openai_organization="your-org-id",  # Optional
)

# Completion model
completion_model = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1000,
)
```

**Available Models:**
- `gpt-4` - Most capable
- `gpt-4-turbo` - Latest GPT-4 with 128k context
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-3.5-turbo-16k` - Extended context version

**Advanced Configuration:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import get_openai_callback

model = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    max_tokens=2000,
    model_kwargs={
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    },
    request_timeout=60,
    max_retries=3,
)

# Track usage/costs
with get_openai_callback() as cb:
    response = model.invoke("Hello, world!")
    print(f"Total cost: ${cb.total_cost:.4f}")
```

### 2. Anthropic Claude

**Installation:**
```bash
pip install langchain-anthropic
```

**Basic Usage:**
```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.7,
    max_tokens=1000,
    anthropic_api_key="your-api-key",  # Optional if in env
)
```

**Available Models:**
- `claude-3-opus-20240229` - Most capable
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast and cost-effective
- `claude-2.1` - Previous generation with 200k context
- `claude-2.0` - Previous generation

**Advanced Configuration:**
```python
model = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0.1,
    max_tokens=4000,
    anthropic_api_key="your-api-key",
    default_request_timeout=60,
    max_retries=3,
)
```

### 3. Google AI (Gemini)

**Installation:**
```bash
pip install langchain-google-genai
```

**Basic Usage:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    max_tokens=1000,
    google_api_key="your-api-key",  # Optional if in env
)
```

**Available Models:**
- `gemini-pro` - General purpose chat model
- `gemini-pro-vision` - Multimodal model with vision

### 4. Azure OpenAI

**Installation:**
```bash
pip install langchain-openai
```

**Configuration:**
```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="your-deployment-name",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000,
)
```

### 5. Hugging Face Hub

**Installation:**
```bash
pip install langchain-huggingface
pip install transformers torch
```

**Configuration:**
```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Using HuggingFace Endpoint (API-based)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_key="your-api-key",
)

# Using local models
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1000,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=pipe)
```

### 6. Cohere

**Installation:**
```bash
pip install langchain-cohere
```

**Configuration:**
```python
from langchain_cohere import ChatCohere

model = ChatCohere(
    model="command",
    temperature=0.7,
    max_tokens=1000,
    cohere_api_key="your-api-key",
)
```

**Available Models:**
- `command` - General purpose model
- `command-light` - Faster, more cost-effective
- `command-nightly` - Latest features (may be less stable)

## Model Comparison

| Provider | Model | Context | Speed | Cost | Best For |
|----------|-------|---------|-------|------|-----------|
| OpenAI | GPT-4 | 8k/128k | Medium | High | Complex reasoning |
| OpenAI | GPT-3.5-Turbo | 4k/16k | Fast | Low | General tasks |
| Anthropic | Claude-3-Opus | 200k | Medium | High | Complex analysis |
| Anthropic | Claude-3-Sonnet | 200k | Fast | Medium | Balanced performance |
| Anthropic | Claude-3-Haiku | 200k | Very Fast | Low | Simple tasks |
| Google | Gemini-Pro | 32k | Fast | Medium | Multimodal tasks |
| Cohere | Command | 4k | Fast | Medium | Business applications |

## Usage Patterns

### 1. Basic Model Invocation

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

# Simple invocation
response = model.invoke("Hello, how are you?")
print(response.content)

# With system message
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
response = model.invoke(messages)
```

### 2. Streaming Responses

```python
# Synchronous streaming
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Asynchronous streaming
async def stream_response():
    async for chunk in model.astream("Tell me a story"):
        print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(stream_response())
```

### 3. Batch Processing

```python
# Process multiple inputs
batch_inputs = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]

responses = model.batch(batch_inputs)
for response in responses:
    print(response.content)
```

### 4. Model Configuration with Callbacks

```python
from langchain_core.callbacks import CallbackHandler

class TokenCounter(CallbackHandler):
    def __init__(self):
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs):
        self.token_count += 1
        print(token, end="", flush=True)

token_counter = TokenCounter()
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[token_counter],
    streaming=True
)

response = model.invoke("Count to 10")
print(f"\nTotal tokens: {token_counter.token_count}")
```

## Error Handling

```python
from langchain_core.exceptions import LangChainException
import openai

def safe_model_call(model, prompt, max_retries=3):
    """Safe model invocation with retry logic."""
    for attempt in range(max_retries):
        try:
            return model.invoke(prompt)
        except openai.RateLimitError:
            print(f"Rate limit hit, attempt {attempt + 1}/{max_retries}")
            import time
            time.sleep(2 ** attempt)  # Exponential backoff
        except openai.APIError as e:
            print(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
        except LangChainException as e:
            print(f"LangChain error: {e}")
            raise

# Usage
response = safe_model_call(model, "Hello, world!")
```

## Cost Optimization

### 1. Choose the Right Model

```python
def select_model(task_complexity: str, budget_level: str):
    """Select appropriate model based on task and budget."""

    if budget_level == "low":
        if task_complexity == "simple":
            return ChatOpenAI(model="gpt-3.5-turbo")
        else:
            return ChatAnthropic(model="claude-3-haiku")

    elif budget_level == "medium":
        if task_complexity == "complex":
            return ChatOpenAI(model="gpt-4")
        else:
            return ChatAnthropic(model="claude-3-sonnet")

    else:  # high budget
        return ChatAnthropic(model="claude-3-opus")

# Usage
model = select_model("complex", "medium")
```

### 2. Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    """Count tokens for a given text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# Estimate costs
text = "Your prompt text here"
token_count = count_tokens(text)
estimated_cost = (token_count / 1000) * 0.002  # GPT-3.5-turbo pricing
print(f"Tokens: {token_count}, Estimated cost: ${estimated_cost:.4f}")
```

## Best Practices

1. **Always handle exceptions** when calling models
2. **Use appropriate models** for your task complexity and budget
3. **Implement rate limiting** for production applications
4. **Cache responses** for repeated queries
5. **Use streaming** for long responses to improve user experience
6. **Monitor costs** and set up alerts if necessary
7. **Set reasonable timeouts** to avoid hanging requests
8. **Use async/await** for better performance in concurrent applications