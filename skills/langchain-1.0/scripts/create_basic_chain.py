#!/usr/bin/env python3
"""
Basic Chain Creation Template for LangChain 1.0

This script provides templates and examples for creating basic LangChain 1.0 chains
using the new LCEL (LangChain Expression Language) syntax.
"""

import os
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def create_simple_chain(
    prompt_template: str,
    model_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7
):
    """
    Create a simple chain with prompt template and model.

    Args:
        prompt_template: Template string for the prompt
        model_provider: Provider name ('openai', 'anthropic')
        model_name: Model name to use
        temperature: Model temperature

    Returns:
        Configured chain ready for invocation
    """
    # Create prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Configure model
    if model_provider.lower() == "openai":
        model = ChatOpenAI(model=model_name, temperature=temperature)
    elif model_provider.lower() == "anthropic":
        model = ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # Create chain using LCEL
    chain = prompt | model | StrOutputParser()

    return chain


def create_chain_with_input_validation(
    system_prompt: str,
    human_prompt: str,
    input_schema: Dict[str, Any],
    model_provider: str = "openai"
):
    """
    Create a chain with input validation and structured prompting.

    Args:
        system_prompt: System instructions
        human_prompt: Human-facing prompt template
        input_schema: Expected input format
        model_provider: Model provider to use

    Returns:
        Chain with input validation
    """
    # Create structured prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # Configure model
    if model_provider.lower() == "openai":
        model = ChatOpenAI()
    elif model_provider.lower() == "anthropic":
        model = ChatAnthropic()
    else:
        model = ChatOpenAI()  # Default fallback

    # Add input validation
    def validate_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input against schema."""
        for key, expected_type in input_schema.items():
            if key not in inputs:
                raise ValueError(f"Missing required input: {key}")
            if not isinstance(inputs[key], expected_type):
                try:
                    inputs[key] = expected_type(inputs[key])
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}")
        return inputs

    # Create chain with validation
    chain = (
        {"validated_input": validate_input}
        | RunnablePassthrough.assign(
            prompt_input=lambda x: x["validated_input"]
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def create_parallel_chain(chain_configs: list):
    """
    Create a chain that runs multiple sub-chains in parallel.

    Args:
        chain_configs: List of chain configuration dictionaries

    Returns:
        Parallel chain configuration
    """
    chains = {}

    for config in chain_configs:
        name = config["name"]
        prompt = config["prompt"]

        # Create individual chain
        chain = create_simple_chain(
            prompt_template=prompt,
            model_provider=config.get("provider", "openai"),
            model_name=config.get("model", "gpt-3.5-turbo")
        )
        chains[name] = chain

    # Create parallel chain
    parallel_chain = RunnableParallel(**chains)

    return parallel_chain


def create_chain_with_memory(
    prompt_template: str,
    model_provider: str = "openai",
    memory_key: str = "chat_history"
):
    """
    Create a chain with conversation memory.

    Args:
        prompt_template: Base prompt template
        model_provider: Model provider
        memory_key: Key for storing conversation history

    Returns:
        Chain configured for memory integration
    """
    # Create prompt that accepts history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    # Configure model
    if model_provider.lower() == "openai":
        model = ChatOpenAI()
    elif model_provider.lower() == "anthropic":
        model = ChatAnthropic()
    else:
        model = ChatOpenAI()

    # Create chain
    chain = prompt | model | StrOutputParser()

    return chain


def create_json_output_chain(
    prompt_template: str,
    output_schema: Dict[str, Any],
    model_provider: str = "openai"
):
    """
    Create a chain that outputs structured JSON.

    Args:
        prompt_template: Prompt template
        output_schema: Expected JSON structure
        model_provider: Model provider

    Returns:
        Chain with JSON output parsing
    """
    # Create prompt with JSON instructions
    json_prompt = f"""{prompt_template}

Please respond with a JSON object using this format:
{output_schema}

Make sure your response is valid JSON only, without any additional text."""

    prompt = ChatPromptTemplate.from_template(json_prompt)

    # Configure model (lower temperature for consistent JSON)
    if model_provider.lower() == "openai":
        model = ChatOpenAI(temperature=0.1)
    elif model_provider.lower() == "anthropic":
        model = ChatAnthropic(temperature=0.1)
    else:
        model = ChatOpenAI(temperature=0.1)

    # Create chain with JSON parser
    chain = prompt | model | JsonOutputParser()

    return chain


# Example usage templates
def example_simple_chain():
    """Example: Simple text generation chain."""
    chain = create_simple_chain(
        prompt_template="Write a {adjective} story about {topic}.",
        model_provider="openai"
    )

    result = chain.invoke({
        "adjective": "short",
        "topic": "a robot learning to paint"
    })
    return result


def example_parallel_chain():
    """Example: Parallel processing chain."""
    chain_configs = [
        {
            "name": "summary",
            "prompt": "Summarize this text in one sentence: {text}",
            "provider": "openai"
        },
        {
            "name": "key_points",
            "prompt": "Extract 3 key points from this text: {text}",
            "provider": "openai"
        },
        {
            "name": "sentiment",
            "prompt": "Analyze the sentiment of this text (positive/negative/neutral): {text}",
            "provider": "openai"
        }
    ]

    chain = create_parallel_chain(chain_configs)

    result = chain.invoke({
        "text": "LangChain 1.0 introduces many exciting features including better performance, improved developer experience, and powerful new abstractions."
    })
    return result


def example_json_chain():
    """Example: JSON output chain."""
    output_schema = {
        "title": "string",
        "summary": "string",
        "tags": ["string"],
        "confidence": "number (0-1)"
    }

    chain = create_json_output_chain(
        prompt_template="Analyze this article and extract information: {article}",
        output_schema=output_schema
    )

    result = chain.invoke({
        "article": "New advances in AI technology are transforming how we work and live."
    })
    return result


if __name__ == "__main__":
    print("🔗 LangChain 1.0 Basic Chain Examples")
    print("=" * 40)

    # Test examples (requires API keys)
    try:
        print("\n1. Simple Chain Example:")
        result = example_simple_chain()
        print(result)

        print("\n2. Parallel Chain Example:")
        result = example_parallel_chain()
        for key, value in result.items():
            print(f"{key}: {value}")

        print("\n3. JSON Output Example:")
        result = example_json_chain()
        print(result)

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Make sure you have API keys configured in .env file")