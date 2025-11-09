#!/usr/bin/env python3
"""
Basic Chain Template for LangChain 1.0

This template provides a complete, ready-to-use example of a basic LangChain 1.0 chain
using LCEL syntax. Copy this file and customize it for your specific use case.
"""

import os
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, SystemMessage

# Model imports - choose your preferred provider
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI


class BasicChainTemplate:
    """
    Template for creating basic LangChain 1.0 chains.

    This class demonstrates various chain patterns and best practices.
    """

    def __init__(
        self,
        model_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the chain template.

        Args:
            model_provider: Provider to use ('openai', 'anthropic', 'google')
            model_name: Specific model to use
            temperature: Model temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the model
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on provider."""
        if self.model_provider.lower() == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.model_provider.lower() == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.model_provider.lower() == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def create_simple_chain(self, template: str):
        """
        Create a simple chain with a prompt template.

        Args:
            template: Prompt template string

        Returns:
            Configured chain ready for invocation
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()
        return chain

    def create_structured_chain(self, system_prompt: str, human_prompt: str):
        """
        Create a chain with structured prompting.

        Args:
            system_prompt: System instructions
            human_prompt: Human-facing prompt template

        Returns:
            Configured chain with structured prompts
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        chain = prompt | self.model | StrOutputParser()
        return chain

    def create_json_chain(self, template: str, output_schema: Dict[str, Any]):
        """
        Create a chain that outputs structured JSON.

        Args:
            template: Prompt template with JSON instructions
            output_schema: Expected JSON structure

        Returns:
            Chain with JSON output parsing
        """
        # Add JSON instructions to template
        json_template = f"""{template}

Please respond with valid JSON using this schema:
{output_schema}

Return ONLY the JSON, no additional text."""

        prompt = ChatPromptTemplate.from_template(json_template)

        # Use lower temperature for consistent JSON
        json_model = self._initialize_model()
        json_model.temperature = 0.1

        chain = prompt | json_model | JsonOutputParser()
        return chain

    def create_parallel_chain(self, chain_configs: list):
        """
        Create a chain that runs multiple operations in parallel.

        Args:
            chain_configs: List of chain configuration dictionaries

        Returns:
            Parallel chain configuration
        """
        chains = {}

        for config in chain_configs:
            name = config["name"]
            template = config["template"]

            # Create individual chain
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model | StrOutputParser()
            chains[name] = chain

        # Create parallel chain
        parallel_chain = RunnableParallel(**chains)
        return parallel_chain

    def create_chain_with_transformation(self, template: str, transformation_func):
        """
        Create a chain with input/output transformation.

        Args:
            template: Prompt template
            transformation_func: Function to transform inputs/outputs

        Returns:
            Chain with transformation step
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnablePassthrough.assign(transformed=transformation_func)
            | prompt
            | self.model
            | StrOutputParser()
        )
        return chain


# Usage Examples
def example_simple_usage():
    """Example: Simple text generation."""

    # Initialize template
    chain_template = BasicChainTemplate(
        model_provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )

    # Create simple chain
    chain = chain_template.create_simple_chain(
        "Write a {adjective} story about {topic}. Keep it under 100 words."
    )

    # Invoke the chain
    result = chain.invoke({
        "adjective": "short",
        "topic": "a robot learning to paint"
    })

    print("Simple Chain Example:")
    print(result)
    return result


def example_structured_prompt():
    """Example: Chain with structured prompting."""

    chain_template = BasicChainTemplate()

    chain = chain_template.create_structured_chain(
        system_prompt="You are a creative writing assistant. Always provide helpful, encouraging feedback.",
        human_prompt="Please help me improve this writing: '{writing}'"
    )

    result = chain.invoke({
        "writing": "The cat sat on the mat. It was bored."
    })

    print("\nStructured Prompt Example:")
    print(result)
    return result


def example_json_output():
    """Example: Chain with JSON output."""

    chain_template = BasicChainTemplate()

    output_schema = {
        "sentiment": "string (positive/negative/neutral)",
        "confidence": "number (0-1)",
        "key_topics": ["string"],
        "summary": "string"
    }

    chain = chain_template.create_json_chain(
        template="Analyze the sentiment and extract key information from this text: {text}",
        output_schema=output_schema
    )

    result = chain.invoke({
        "text": "I'm really excited about the new features in LangChain 1.0! The LCEL syntax is amazing and makes development so much easier."
    })

    print("\nJSON Output Example:")
    print(result)
    return result


def example_parallel_processing():
    """Example: Parallel chain processing."""

    chain_template = BasicChainTemplate()

    chain_configs = [
        {
            "name": "summary",
            "template": "Summarize this in one sentence: {text}"
        },
        {
            "name": "sentiment",
            "template": "What is the sentiment (positive/negative/neutral)? {text}"
        },
        {
            "name": "keywords",
            "template": "Extract 3 key topics: {text}"
        }
    ]

    chain = chain_template.create_parallel_chain(chain_configs)

    result = chain.invoke({
        "text": "LangChain 1.0 introduces powerful new features including LCEL syntax, improved performance, and better developer experience."
    })

    print("\nParallel Processing Example:")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def example_with_transformation():
    """Example: Chain with input transformation."""

    chain_template = BasicChainTemplate()

    def enhance_prompt(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform inputs to add context."""
        enhanced = inputs.copy()
        enhanced["full_context"] = f"Topic: {inputs['topic']}\nAudience: {inputs['audience']}\nStyle: {inputs['style']}"
        return enhanced

    chain = chain_template.create_chain_with_transformation(
        template="Create content about: {full_context}",
        transformation_func=enhance_prompt
    )

    result = chain.invoke({
        "topic": "artificial intelligence",
        "audience": "beginners",
        "style": "educational and friendly"
    })

    print("\nTransformation Example:")
    print(result)
    return result


def example_error_handling():
    """Example: Chain with proper error handling."""

    chain_template = BasicChainTemplate()
    chain = chain_template.create_simple_chain("Respond to: {input}")

    try:
        result = chain.invoke({"input": "Hello, world!"})
        print("\nError Handling Example - Success:")
        print(result)
    except Exception as e:
        print(f"\nError Handling Example - Failed: {e}")
        # Implement fallback logic here
        return "Sorry, I encountered an error. Please try again."


def example_streaming():
    """Example: Streaming chain responses."""

    chain_template = BasicChainTemplate()
    chain = chain_template.create_simple_chain("Tell me a short story about {topic}")

    print("\nStreaming Example:")
    print("Story: ", end="", flush=True)

    try:
        for chunk in chain.stream({"topic": "time travel"}):
            print(chunk.content, end="", flush=True)
        print()  # New line at the end
    except Exception as e:
        print(f"Streaming failed: {e}")


if __name__ == "__main__":
    """Run all examples to demonstrate different chain patterns."""

    print("🔗 LangChain 1.0 Basic Chain Template Examples")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("💡 Set your API key to run these examples: export OPENAI_API_KEY='your-key'")
        exit(1)

    try:
        # Run examples
        example_simple_usage()
        example_structured_prompt()
        example_json_output()
        example_parallel_processing()
        example_with_transformation()
        example_error_handling()
        example_streaming()

        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys configured")
        print("   - Required packages installed")
        print("   - Internet connection")