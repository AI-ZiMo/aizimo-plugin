#!/usr/bin/env python3
"""
Output Parser Templates for LangChain 1.0

This script provides templates and examples for various output parsers
to structure LLM responses in specific formats.
"""

import json
import re
from typing import List, Dict, Any, Optional, Type, Union
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    BaseOutputParser
)
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI


class CustomOutputParser(BaseOutputParser):
    """Custom output parser for specific formats."""

    def __init__(self, pattern: str, format_description: str):
        """
        Initialize custom parser.

        Args:
            pattern: Regex pattern to match output
            format_description: Description of expected format
        """
        super().__init__()
        self.pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        self.format_description = format_description

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse text according to the pattern."""
        match = self.pattern.search(text)
        if not match:
            raise OutputParserException(f"Output doesn't match expected format: {self.format_description}")
        return match.groupdict()

    def get_format_instructions(self) -> str:
        """Get instructions for the format."""
        return f"Output must match this format: {self.format_description}"


class StructuredDataParser(BaseOutputParser):
    """Parser for structured data like key-value pairs."""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse key-value pairs from text."""
        data = {}
        # Match key: value pairs
        pattern = r"(\w+(?:\s+\w+)*):\s*(.+?)(?=\n\w+:|$)"
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

        for key, value in matches:
            # Clean up the key and value
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            # Try to convert to appropriate type
            if value.lower() in ["true", "false"]:
                data[key] = value.lower() == "true"
            elif value.isdigit():
                data[key] = int(value)
            elif self._is_float(value):
                data[key] = float(value)
            else:
                data[key] = value

        return data

    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False


# Pydantic models for structured output
class MovieReview(BaseModel):
    """Structured model for movie reviews."""
    title: str = Field(description="The title of the movie")
    rating: float = Field(description="Rating from 0-10", ge=0, le=10)
    genre: List[str] = Field(description="List of genres")
    summary: str = Field(description="Brief summary of the plot")
    recommendation: str = Field(description="Would you recommend it? (yes/no/maybe)")

    @validator('rating')
    def rating_must_be_valid(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Rating must be between 0 and 10')
        return v


class ProductAnalysis(BaseModel):
    """Structured model for product analysis."""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    features: List[str] = Field(description="Key features")
    pros: List[str] = Field(description="Advantages")
    cons: List[str] = Field(description="Disadvantages")
    overall_score: float = Field(description="Overall score 0-100", ge=0, le=100)


class CodeAnalysis(BaseModel):
    """Structured model for code analysis."""
    language: str = Field(description="Programming language")
    complexity: str = Field(description="Complexity level (low/medium/high)")
    issues: List[str] = Field(description="Issues found")
    suggestions: List[str] = Field(description="Improvement suggestions")
    security_score: float = Field(description="Security score 0-10", ge=0, le=10)


class OutputParserTemplates:
    """
    Collection of output parser templates and examples.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize parser templates.

        Args:
            model_name: Model to use for parsing
        """
        self.model = ChatOpenAI(model=model_name, temperature=0.1)

    def create_json_parser_chain(self, prompt_template: str):
        """
        Create a chain with JSON output parsing.

        Args:
            prompt_template: Template with JSON instructions

        Returns:
            Chain with JSON parser
        """
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(
            f"{prompt_template}\n\n{parser.get_format_instructions()}"
        )
        chain = prompt | self.model | parser
        return chain

    def create_pydantic_parser_chain(self, prompt_template: str, pydantic_model: Type[BaseModel]):
        """
        Create a chain with Pydantic output parsing.

        Args:
            prompt_template: Template for the prompt
            pydantic_model: Pydantic model class

        Returns:
            Chain with Pydantic parser
        """
        parser = PydanticOutputParser(pydantic_model=pydantic_model)
        prompt = ChatPromptTemplate.from_template(
            f"{prompt_template}\n\n{parser.get_format_instructions()}"
        )
        chain = prompt | self.model | parser
        return chain

    def create_list_parser_chain(self, prompt_template: str):
        """
        Create a chain that outputs a comma-separated list.

        Args:
            prompt_template: Template for the prompt

        Returns:
            Chain with list parser
        """
        parser = CommaSeparatedListOutputParser()
        prompt = ChatPromptTemplate.from_template(
            f"{prompt_template}\n\n{parser.get_format_instructions()}"
        )
        chain = prompt | self.model | parser
        return chain

    def create_custom_parser_chain(self, prompt_template: str, pattern: str, description: str):
        """
        Create a chain with custom output parsing.

        Args:
            prompt_template: Template for the prompt
            pattern: Regex pattern for parsing
            description: Description of expected format

        Returns:
            Chain with custom parser
        """
        parser = CustomOutputParser(pattern, description)
        prompt = ChatPromptTemplate.from_template(
            f"{prompt_template}\n\n{parser.get_format_instructions()}"
        )
        chain = prompt | self.model | parser
        return chain

    def create_structured_data_chain(self, prompt_template: str):
        """
        Create a chain that outputs structured key-value data.

        Args:
            prompt_template: Template for the prompt

        Returns:
            Chain with structured data parser
        """
        parser = StructuredDataParser()
        prompt = ChatPromptTemplate.from_template(
            f"{prompt_template}\n\nFormat your response as key-value pairs like:\n"
            "Key: Value\nAnother Key: Another Value\netc."
        )
        chain = prompt | self.model | parser
        return chain

    def create_multi_format_chain(self, prompt_template: str):
        """
        Create a chain that can output in multiple formats.

        Args:
            prompt_template: Template with format selection

        Returns:
            Chain that handles multiple output formats
        """
        def multi_format_parser(text: str) -> Dict[str, Any]:
            """Parse text that could be in different formats."""
            text = text.strip()

            # Try JSON first
            try:
                return {"format": "json", "data": json.loads(text)}
            except json.JSONDecodeError:
                pass

            # Try key-value pairs
            try:
                structured_parser = StructuredDataParser()
                data = structured_parser.parse(text)
                if data:
                    return {"format": "structured", "data": data}
            except OutputParserException:
                pass

            # Try comma-separated list
            if "," in text:
                items = [item.strip() for item in text.split(",")]
                return {"format": "list", "data": items}

            # Default to plain text
            return {"format": "text", "data": text}

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.model | StrOutputParser() | multi_format_parser
        return chain


def example_json_parsing():
    """Example: JSON output parsing."""
    print("📄 JSON Output Parsing Example")
    print("-" * 35)

    templates = OutputParserTemplates()

    chain = templates.create_json_parser_chain(
        "Analyze this movie and provide details: {movie_title}"
    )

    result = chain.invoke({"movie_title": "The Matrix"})
    print(f"Result: {json.dumps(result, indent=2)}")


def example_pydantic_parsing():
    """Example: Pydantic model parsing."""
    print("\n📄 Pydantic Output Parsing Example")
    print("-" * 40)

    templates = OutputParserTemplates()

    # Movie review example
    chain = templates.create_pydantic_parser_chain(
        "Review this movie: {movie_title}",
        MovieReview
    )

    result = chain.invoke({"movie_title": "Inception"})
    print(f"Movie Review: {result}")
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}/10")
    print(f"Genres: {', '.join(result.genre)}")
    print(f"Summary: {result.summary[:100]}{'...' if len(result.summary) > 100 else ''}")

    # Product analysis example
    product_chain = templates.create_pydantic_parser_chain(
        "Analyze this product: {product_description}",
        ProductAnalysis
    )

    product_result = product_chain.invoke({
        "product_description": "Wireless headphones with noise cancellation, 20-hour battery life, Bluetooth 5.0"
    })
    print(f"\nProduct Analysis: {product_result.name}")
    print(f"Price estimate: ${product_result.price}")
    print(f"Overall score: {product_result.overall_score}/100")


def example_list_parsing():
    """Example: List output parsing."""
    print("\n📄 List Output Parsing Example")
    print("-" * 33)

    templates = OutputParserTemplates()

    chain = templates.create_list_parser_chain(
        "List 5 important concepts in {subject}"
    )

    result = chain.invoke({"subject": "machine learning"})
    print(f"Machine Learning Concepts: {result}")
    for i, concept in enumerate(result, 1):
        print(f"  {i}. {concept}")


def example_custom_parsing():
    """Example: Custom format parsing."""
    print("\n📄 Custom Output Parsing Example")
    print("-" * 36)

    templates = OutputParserTemplates()

    # Email format parser
    email_pattern = r"From:\s*(?P<from>.+?)\nTo:\s*(?P<to>.+?)\nSubject:\s*(?P<subject>.+?)\n\n(?P<body>.+)"
    chain = templates.create_custom_parser_chain(
        "Generate a professional email about {topic}",
        email_pattern,
        "Format: From: [sender]\nTo: [recipient]\nSubject: [subject]\n\n[body]"
    )

    result = chain.invoke({"topic": "project update"})
    print(f"Email parsed:")
    print(f"  From: {result['from']}")
    print(f"  To: {result['to']}")
    print(f"  Subject: {result['subject']}")
    print(f"  Body: {result['body'][:100]}{'...' if len(result['body']) > 100 else ''}")


def example_structured_data_parsing():
    """Example: Structured key-value parsing."""
    print("\n📄 Structured Data Parsing Example")
    print("-" * 38)

    templates = OutputParserTemplates()

    chain = templates.create_structured_data_chain(
        "Analyze this code snippet and provide insights: {code_snippet}"
    )

    code_snippet = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
    """

    result = chain.invoke({"code_snippet": code_snippet})
    print(f"Code Analysis Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def example_code_analysis():
    """Example: Code analysis with Pydantic model."""
    print("\n📄 Code Analysis Example")
    print("-" * 28)

    templates = OutputParserTemplates()

    chain = templates.create_pydantic_parser_chain(
        "Analyze this code for security and quality: {code_snippet}",
        CodeAnalysis
    )

    vulnerable_code = """
import pickle
import os

def load_user_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def execute_command(cmd):
    os.system(cmd)
    """

    result = chain.invoke({"code_snippet": vulnerable_code})
    print(f"Language: {result.language}")
    print(f"Complexity: {result.complexity}")
    print(f"Security Score: {result.security_score}/10")
    print("Issues found:")
    for issue in result.issues:
        print(f"  - {issue}")
    print("Suggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")


def example_multi_format_output():
    """Example: Multi-format output handling."""
    print("\n📄 Multi-Format Output Example")
    print("-" * 34)

    templates = OutputParserTemplates()

    chain = templates.create_multi_format_chain(
        "Provide information about {topic} in your preferred format"
    )

    topics = ["Python programming", "Artificial Intelligence", "Data Science"]

    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"\nTopic: {topic}")
        print(f"Format: {result['format']}")
        if result['format'] == 'json':
            print(f"Data: {json.dumps(result['data'], indent=2)}")
        elif result['format'] == 'list':
            print(f"Data: {result['data']}")
        else:
            print(f"Data: {result['data'][:100]}{'...' if len(str(result['data'])) > 100 else ''}")


def example_error_handling():
    """Example: Error handling with output parsers."""
    print("\n📄 Error Handling Example")
    print("-" * 29)

    templates = OutputParserTemplates()

    # Create a chain that might fail
    chain = templates.create_json_parser_chain(
        "Provide structured data about: {topic}"
    )

    test_inputs = [
        "A simple cat",
        "Complex information about quantum computing",
        "Invalid response that breaks JSON formatting"
    ]

    for test_input in test_inputs:
        try:
            print(f"\nTesting: {test_input}")
            result = chain.invoke({"topic": test_input})
            print(f"✅ Success: {result}")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")

            # Fallback strategy
            fallback_chain = templates.create_structured_data_chain(
                "Provide structured information about: {topic}"
            )
            try:
                fallback_result = fallback_chain.invoke({"topic": test_input})
                print(f"🔄 Fallback success: {fallback_result}")
            except Exception as fallback_error:
                print(f"❌ Fallback failed: {fallback_error}")


if __name__ == "__main__":
    """Run output parser examples."""
    print("📄 LangChain 1.0 Output Parser Examples")
    print("=" * 45)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key'")
        exit(1)

    try:
        # Run examples
        example_json_parsing()
        example_pydantic_parsing()
        example_list_parsing()
        example_custom_parsing()
        example_structured_data_parsing()
        example_code_analysis()
        example_multi_format_output()
        example_error_handling()

        print("\n" + "=" * 45)
        print("✅ All output parser examples completed!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys configured")
        print("   - Required packages installed")
        print("   - Internet connection")