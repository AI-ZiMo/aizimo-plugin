#!/usr/bin/env python3
"""
Message Handling Examples for LangChain 1.0

This script demonstrates practical examples of working with AIMessage objects,
including metadata extraction, cost calculation, and response analysis.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Optional: for cost calculation
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    get_openai_callback = None


class MessageHandler:
    """
    Utility class for handling and analyzing AIMessage objects.
    """

    def __init__(self):
        self.conversation_history = []
        self.total_tokens_used = 0
        self.estimated_cost = 0.0

    def extract_message_info(self, message: AIMessage) -> Dict[str, Any]:
        """
        Extract comprehensive information from an AIMessage object.

        Args:
            message: The AIMessage object to analyze

        Returns:
            Dictionary containing extracted information
        """
        info = {
            "basic": {
                "content": message.content,
                "message_id": message.id,
                "content_length": len(message.content) if message.content else 0,
                "word_count": len(message.content.split()) if message.content else 0
            },
            "metadata": {
                "model_provider": message.response_metadata.get("model_provider"),
                "model_name": message.response_metadata.get("model_name"),
                "finish_reason": message.response_metadata.get("finish_reason"),
                "system_fingerprint": message.response_metadata.get("system_fingerprint")
            },
            "token_usage": self._extract_token_usage(message),
            "additional_info": {
                "has_refusal": message.additional_kwargs.get("refusal") is not None,
                "refusal_content": message.additional_kwargs.get("refusal"),
                "provider_response_id": message.response_metadata.get("id")
            }
        }

        return info

    def _extract_token_usage(self, message: AIMessage) -> Dict[str, Any]:
        """
        Extract token usage information from different metadata sources.
        """
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "source": "unknown"
        }

        # Try usage_metadata first (standardized format)
        if message.usage_metadata:
            token_usage.update({
                "input_tokens": message.usage_metadata.get("input_tokens", 0),
                "output_tokens": message.usage_metadata.get("output_tokens", 0),
                "total_tokens": message.usage_metadata.get("total_tokens", 0),
                "source": "usage_metadata"
            })

        # Fallback to response_metadata for some providers
        elif "token_usage" in message.response_metadata:
            usage = message.response_metadata["token_usage"]
            token_usage.update({
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0),
                "source": "response_metadata"
            })

        return token_usage

    def calculate_cost(self, message: AIMessage, provider: str = "openai") -> Dict[str, Any]:
        """
        Calculate estimated cost based on token usage.

        Args:
            message: AIMessage object
            provider: Provider name for pricing

        Returns:
            Cost calculation details
        """
        token_usage = self._extract_token_usage(message)

        if token_usage["total_tokens"] == 0:
            return {"error": "No token usage information available"}

        # Pricing per 1K tokens (example rates - update with current pricing)
        pricing_data = {
            "openai": {
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03}
            },
            "anthropic": {
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-opus": {"input": 0.015, "output": 0.075}
            },
            "deepseek": {
                "deepseek-chat": {"input": 0.0001, "output": 0.0002}
            }
        }

        model_name = message.response_metadata.get("model_name", "gpt-3.5-turbo")
        provider_pricing = pricing_data.get(provider, {})
        model_pricing = provider_pricing.get(model_name, {"input": 0.001, "output": 0.002})

        input_cost = (token_usage["input_tokens"] / 1000) * model_pricing["input"]
        output_cost = (token_usage["output_tokens"] / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        cost_info = {
            "provider": provider,
            "model": model_name,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
            "total_tokens": token_usage["total_tokens"],
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "pricing_source": "estimated"
        }

        self.total_tokens_used += token_usage["total_tokens"]
        self.estimated_cost += total_cost

        return cost_info

    def analyze_response_quality(self, message: AIMessage) -> Dict[str, Any]:
        """
        Analyze response quality metrics.

        Args:
            message: AIMessage object to analyze

        Returns:
            Quality analysis metrics
        """
        content = message.content or ""
        token_usage = self._extract_token_usage(message)

        analysis = {
            "content_metrics": {
                "character_count": len(content),
                "word_count": len(content.split()),
                "sentence_count": content.count('.') + content.count('!') + content.count('?'),
                "avg_words_per_sentence": 0
            },
            "efficiency_metrics": {
                "tokens_per_word": 0,
                "words_per_token": 0,
                "characters_per_token": 0
            },
            "completion_status": {
                "is_complete": message.response_metadata.get("finish_reason") == "stop",
                "was_filtered": message.response_metadata.get("finish_reason") == "content_filter",
                "hit_length_limit": message.response_metadata.get("finish_reason") == "length",
                "has_refusal": message.additional_kwargs.get("refusal") is not None
            }
        }

        # Calculate efficiency metrics
        if token_usage["output_tokens"] > 0:
            word_count = analysis["content_metrics"]["word_count"]
            char_count = analysis["content_metrics"]["character_count"]
            output_tokens = token_usage["output_tokens"]

            analysis["efficiency_metrics"].update({
                "tokens_per_word": output_tokens / word_count if word_count > 0 else 0,
                "words_per_token": word_count / output_tokens if output_tokens > 0 else 0,
                "characters_per_token": char_count / output_tokens if output_tokens > 0 else 0
            })

        # Calculate words per sentence
        sentence_count = analysis["content_metrics"]["sentence_count"]
        word_count = analysis["content_metrics"]["word_count"]
        if sentence_count > 0:
            analysis["content_metrics"]["avg_words_per_sentence"] = word_count / sentence_count

        return analysis

    def add_to_conversation(self, message: AIMessage, role: str = "assistant"):
        """
        Add message to conversation history.

        Args:
            message: The AIMessage object
            role: Role in conversation (assistant, system, etc.)
        """
        self.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "extracted_info": self.extract_message_info(message)
        })

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get overall conversation statistics.

        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {"error": "No conversation history"}

        total_messages = len([msg for msg in self.conversation_history if msg["role"] == "assistant"])
        total_tokens = sum(msg["extracted_info"]["token_usage"]["total_tokens"]
                          for msg in self.conversation_history if msg["role"] == "assistant")
        total_words = sum(msg["extracted_info"]["basic"]["word_count"]
                         for msg in self.conversation_history if msg["role"] == "assistant")

        stats = {
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "average_tokens_per_message": total_tokens / total_messages if total_messages > 0 else 0,
            "average_words_per_message": total_words / total_messages if total_messages > 0 else 0,
            "total_estimated_cost": round(self.estimated_cost, 6),
            "conversation_duration": None
        }

        # Calculate duration if timestamps are available
        if len(self.conversation_history) >= 2:
            first_time = datetime.fromisoformat(self.conversation_history[0]["timestamp"])
            last_time = datetime.fromisoformat(self.conversation_history[-1]["timestamp"])
            duration = last_time - first_time
            stats["conversation_duration"] = str(duration)

        return stats

    def save_conversation(self, filename: str):
        """
        Save conversation history to JSON file.

        Args:
            filename: Path to save the conversation
        """
        serializable_history = []

        for item in self.conversation_history:
            serializable_item = {
                "role": item["role"],
                "timestamp": item["timestamp"],
                "extracted_info": item["extracted_info"]
            }

            # Serialize message content
            if hasattr(item["message"], "content"):
                serializable_item["content"] = item["message"].content

            serializable_history.append(serializable_item)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation": serializable_history,
                "stats": self.get_conversation_stats(),
                "saved_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

        print(f"Conversation saved to {filename}")


# Example usage functions
def example_basic_message_handling():
    """
    Demonstrate basic AIMessage handling.
    """
    print("🔍 Basic Message Handling Example")
    print("=" * 40)

    # Initialize model and handler
    model = ChatOpenAI(model="gpt-3.5-turbo")
    handler = MessageHandler()

    # Create a simple prompt and get response
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    chain = prompt | model

    try:
        response = chain.invoke({"topic": "programming"})

        print("📝 Raw response type:", type(response).__name__)
        print("📄 Response content preview:", response.content[:100] + "..." if len(response.content) > 100 else response.content)
        print()

        # Extract and display information
        info = handler.extract_message_info(response)

        print("📊 Message Information:")
        print(f"  Content length: {info['basic']['content_length']} characters")
        print(f"  Word count: {info['basic']['word_count']} words")
        print(f"  Message ID: {info['basic']['message_id']}")
        print()

        print("🤖 Model Information:")
        print(f"  Provider: {info['metadata']['model_provider']}")
        print(f"  Model: {info['metadata']['model_name']}")
        print(f"  Finish reason: {info['metadata']['finish_reason']}")
        print()

        print("🪙 Token Usage:")
        token_usage = info["token_usage"]
        print(f"  Input tokens: {token_usage['input_tokens']}")
        print(f"  Output tokens: {token_usage['output_tokens']}")
        print(f"  Total tokens: {token_usage['total_tokens']}")
        print(f"  Data source: {token_usage['source']}")
        print()

        # Calculate cost
        cost_info = handler.calculate_cost(response)
        print("💰 Cost Analysis:")
        print(f"  Estimated cost: ${cost_info['total_cost']}")
        print(f"  Model: {cost_info['model']}")
        print()

        # Analyze quality
        quality = handler.analyze_response_quality(response)
        print("📈 Quality Analysis:")
        print(f"  Words per sentence: {quality['content_metrics']['avg_words_per_sentence']:.1f}")
        print(f"  Words per token: {quality['efficiency_metrics']['words_per_token']:.2f}")
        print(f"  Response complete: {quality['completion_status']['is_complete']}")

        # Add to conversation
        handler.add_to_conversation(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have OPENAI_API_KEY set in your environment")


def example_conversation_tracking():
    """
    Demonstrate conversation tracking across multiple messages.
    """
    print("\n🗣️  Conversation Tracking Example")
    print("=" * 40)

    model = ChatOpenAI(model="gpt-3.5-turbo")
    handler = MessageHandler()

    # Simulate a conversation
    conversation_prompts = [
        "Hello! My name is Alex. What's your name?",
        "I'm interested in learning about artificial intelligence. Can you give me a brief introduction?",
        "What are the main types of machine learning?",
        "Can you recommend some resources for beginners?"
    ]

    for i, prompt_text in enumerate(conversation_prompts, 1):
        print(f"\n👤 User ({i}): {prompt_text}")

        try:
            response = model.invoke(prompt_text)
            print(f"🤖 Assistant: {response.content[:100]}...")

            # Process the response
            handler.add_to_conversation(response)

            # Show immediate stats
            info = handler.extract_message_info(response)
            print(f"   📊 Tokens used: {info['token_usage']['total_tokens']}")

        except Exception as e:
            print(f"❌ Error on turn {i}: {e}")

    # Show final conversation stats
    print("\n📊 Final Conversation Statistics:")
    stats = handler.get_conversation_stats()

    if "error" not in stats:
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Average tokens per message: {stats['average_tokens_per_message']:.1f}")
        print(f"  Total estimated cost: ${stats['total_estimated_cost']}")
        print(f"  Duration: {stats.get('conversation_duration', 'N/A')}")

    # Save conversation
    handler.save_conversation("example_conversation.json")


def example_provider_comparison():
    """
    Compare message structures across different providers.
    """
    print("\n🔄 Provider Comparison Example")
    print("=" * 40)

    handler = MessageHandler()
    providers = []

    # Test with OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_model = ChatOpenAI(model="gpt-3.5-turbo")
            response = openai_model.invoke("Say hello in one sentence")

            info = handler.extract_message_info(response)
            providers.append({
                "name": "OpenAI GPT-3.5-turbo",
                "info": info
            })
            print(f"✅ OpenAI response received: {response.content[:50]}...")
        except Exception as e:
            print(f"❌ OpenAI error: {e}")

    # Test with Anthropic if available
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            anthropic_model = ChatAnthropic(model="claude-3-haiku")
            response = anthropic_model.invoke("Say hello in one sentence")

            info = handler.extract_message_info(response)
            providers.append({
                "name": "Anthropic Claude-3-haiku",
                "info": info
            })
            print(f"✅ Anthropic response received: {response.content[:50]}...")
        except Exception as e:
            print(f"❌ Anthropic error: {e}")

    # Compare provider responses
    if len(providers) > 1:
        print("\n📊 Provider Comparison:")
        for provider in providers:
            print(f"\n🏢 {provider['name']}:")
            info = provider["info"]
            print(f"  Model: {info['metadata']['model_name']}")
            print(f"  Tokens: {info['token_usage']['total_tokens']}")
            print(f"  Finish reason: {info['metadata']['finish_reason']}")
            print(f"  Content length: {info['basic']['content_length']}")
    elif len(providers) == 1:
        print("\n💡 Only one provider available for comparison")
    else:
        print("\n⚠️  No providers available. Set API keys to test.")


if __name__ == "__main__":
    """
    Run all message handling examples.
    """
    print("🔧 LangChain 1.0 Message Handling Examples")
    print("=" * 50)

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        print("⚠️  Warning: No API keys found!")
        print("💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run examples")
        exit(1)

    print(f"🔑 Available providers: {', '.join(filter(None, ['OpenAI' if has_openai else None, 'Anthropic' if has_anthropic else None])))}")

    try:
        # Run examples
        example_basic_message_handling()
        example_conversation_tracking()
        example_provider_comparison()

        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Check your API keys and internet connection")