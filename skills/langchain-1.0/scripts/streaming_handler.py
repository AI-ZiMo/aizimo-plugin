#!/usr/bin/env python3
"""
Streaming Handler Templates for LangChain 1.0

This script provides templates and examples for implementing streaming responses
in LangChain 1.0 applications.
"""

import asyncio
import time
from typing import AsyncIterator, Iterator, Dict, Any, Optional, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for streaming responses with additional features.
    """

    def __init__(self, on_token: Optional[Callable[[str], None]] = None):
        """
        Initialize streaming callback handler.

        Args:
            on_token: Optional callback function for each token
        """
        super().__init__()
        self.on_token = on_token
        self.tokens = []
        self.start_time = None
        self.end_time = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
        if self.on_token:
            self.on_token(token)
        print(token, end="", flush=True)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs) -> None:
        """Called when LLM starts processing."""
        self.start_time = time.time()
        self.tokens = []

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes processing."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        total_tokens = len(self.tokens)
        print(f"\n\n[Streaming completed: {total_tokens} tokens in {duration:.2f}s]")

    def get_full_response(self) -> str:
        """Get the complete response."""
        return "".join(self.tokens)

    def get_token_count(self) -> int:
        """Get total number of tokens."""
        return len(self.tokens)

    def get_duration(self) -> float:
        """Get processing duration."""
        return self.end_time - self.start_time if self.start_time and self.end_time else 0


class StreamingChatManager:
    """
    Manager for streaming chat functionality with enhanced features.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize streaming chat manager.

        Args:
            model_name: Model to use
            temperature: Model temperature
            max_tokens: Maximum tokens in response
        """
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True
        )
        self.prompt = ChatPromptTemplate.from_template("{input}")
        self.chain = self.prompt | self.model | StrOutputParser()

    def stream_response(self, input_text: str) -> Iterator[str]:
        """
        Stream response synchronously.

        Args:
            input_text: Input text to process

        Yields:
            Response chunks
        """
        for chunk in self.chain.stream({"input": input_text}):
            if chunk.content:
                yield chunk.content

    async def astream_response(self, input_text: str) -> AsyncIterator[str]:
        """
        Stream response asynchronously.

        Args:
            input_text: Input text to process

        Yields:
            Response chunks
        """
        async for chunk in self.chain.astream({"input": input_text}):
            if chunk.content:
                yield chunk.content

    def stream_with_callback(self, input_text: str, callback: StreamingCallbackHandler):
        """
        Stream response with custom callback.

        Args:
            input_text: Input text to process
            callback: Callback handler instance

        Returns:
            Complete response
        """
        chain_with_callback = self.prompt | self.model | StrOutputParser()
        return chain_with_callback.invoke(
            {"input": input_text},
            config={"callbacks": [callback]}
        )

    def stream_with_analysis(self, input_text: str):
        """
        Stream response with real-time analysis.

        Args:
            input_text: Input text to process

        Yields:
            Tuple of (chunk, analysis_data)
        """
        token_count = 0
        word_count = 0
        start_time = time.time()

        for chunk in self.chain.stream({"input": input_text}):
            if chunk.content:
                token_count += 1
                words = chunk.content.split()
                word_count += len(words)
                current_time = time.time()
                elapsed_time = current_time - start_time

                analysis = {
                    "token_count": token_count,
                    "word_count": word_count,
                    "elapsed_time": elapsed_time,
                    "tokens_per_second": token_count / elapsed_time if elapsed_time > 0 else 0
                }

                yield chunk.content, analysis


class MultiModelStreamer:
    """
    Stream responses from multiple models simultaneously.
    """

    def __init__(self, models: Dict[str, ChatOpenAI]):
        """
        Initialize multi-model streamer.

        Args:
            models: Dictionary of model instances
        """
        self.models = models
        self.prompts = {name: ChatPromptTemplate.from_template("{input}") for name in models}
        self.chains = {
            name: prompt | model | StrOutputParser()
            for name, (prompt, model) in zip(models, zip(self.prompts.values(), models.values()))
        }

    async def stream_multiple_models(self, input_text: str) -> AsyncIterator[Dict[str, str]]:
        """
        Stream responses from multiple models concurrently.

        Args:
            input_text: Input text to process

        Yields:
            Dictionary with model names as keys and chunks as values
        """
        async def get_stream_generator(model_name: str, chain):
            async for chunk in chain.astream({"input": input_text}):
                if chunk.content:
                    yield model_name, chunk.content

        # Create async generators for each model
        generators = [
            get_stream_generator(name, chain)
            for name, chain in self.chains.items()
        ]

        # Process all streams concurrently
        while generators:
            # Wait for any generator to produce a chunk
            tasks = [gen.__anext__() for gen in generators]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Process completed tasks
            result = {}
            for task in done:
                try:
                    model_name, chunk = task.result()
                    result[model_name] = chunk
                except StopAsyncIteration:
                    # Remove completed generator
                    gen_index = tasks.index(task)
                    generators.pop(gen_index)

            if result:
                yield result

            # Remove completed generators from the list
            generators = [gen for gen in generators if gen not in [task for task in done]]


class BufferedStreamer:
    """
    Stream responses with buffering for smoother output.
    """

    def __init__(self, base_streamer, buffer_size: int = 50, buffer_timeout: float = 0.1):
        """
        Initialize buffered streamer.

        Args:
            base_streamer: Base streaming chain
            buffer_size: Maximum buffer size before flushing
            buffer_timeout: Maximum time before flushing buffer
        """
        self.base_streamer = base_streamer
        self.buffer_size = buffer_size
        self.buffer_timeout = buffer_timeout

    def stream_buffered(self, input_text: str) -> Iterator[str]:
        """
        Stream response with buffering.

        Args:
            input_text: Input text to process

        Yields:
            Buffered chunks
        """
        buffer = ""
        last_flush_time = time.time()

        for chunk in self.base_streamer.stream({"input": input_text}):
            if chunk.content:
                buffer += chunk.content
                current_time = time.time()

                # Flush conditions
                should_flush = (
                    len(buffer) >= self.buffer_size or
                    (current_time - last_flush_time) >= self.buffer_timeout or
                    chunk.content.endswith(('.', '!', '?', '\n'))
                )

                if should_flush and buffer:
                    yield buffer
                    buffer = ""
                    last_flush_time = current_time

        # Flush any remaining content
        if buffer:
            yield buffer


def example_basic_streaming():
    """Example: Basic streaming functionality."""
    print("🌊 Basic Streaming Example")
    print("-" * 30)

    streamer = StreamingChatManager()

    print("Question: Tell me a story about a robot discovering music")
    print("Response: ", end="", flush=True)

    for chunk in streamer.stream_response("Tell me a short story about a robot discovering music"):
        # Already printed in the stream
        pass

    print("\n")


async def example_async_streaming():
    """Example: Async streaming functionality."""
    print("\n🌊 Async Streaming Example")
    print("-" * 28)

    streamer = StreamingChatManager()

    print("Question: What are the benefits of renewable energy?")
    print("Response: ", end="", flush=True)

    async for chunk in streamer.astream_response("What are the benefits of renewable energy?"):
        print(chunk, end="", flush=True)

    print("\n")


def example_callback_streaming():
    """Example: Streaming with custom callbacks."""
    print("\n🌊 Callback Streaming Example")
    print("-" * 32)

    streamer = StreamingChatManager()

    # Create custom callback
    def on_token(token: str):
        # Custom token processing
        if '\n' in token:
            print(f"\n[New line detected at {time.strftime('%H:%M:%S')}]")

    callback = StreamingCallbackHandler(on_token=on_token)

    print("Question: Explain quantum computing simply")
    print("Response: ", end="", flush=True)

    response = streamer.stream_with_callback(
        "Explain quantum computing in simple terms",
        callback
    )

    print(f"\nFull response length: {len(callback.get_full_response())} characters")
    print(f"Total tokens: {callback.get_token_count()}")


def example_analyzed_streaming():
    """Example: Streaming with real-time analysis."""
    print("\n🌊 Analyzed Streaming Example")
    print("-" * 32)

    streamer = StreamingChatManager()

    print("Question: Describe the process of photosynthesis")
    print("Response with analysis:", end="", flush=True)

    for chunk, analysis in streamer.stream_with_analysis(
        "Describe the process of photosynthesis in detail"
    ):
        print(chunk, end="", flush=True)

        # Show analysis every 50 tokens
        if analysis["token_count"] % 50 == 0:
            print(f"\n[Analysis: {analysis['token_count']} tokens, "
                  f"{analysis['tokens_per_second']:.1f} tokens/sec]")

    print("\n")


async def example_multi_model_streaming():
    """Example: Streaming from multiple models."""
    print("\n🌊 Multi-Model Streaming Example")
    print("-" * 34)

    models = {
        "gpt-3.5": ChatOpenAI(model="gpt-3.5-turbo", streaming=True),
        "gpt-4": ChatOpenAI(model="gpt-4", streaming=True)
    }

    streamer = MultiModelStreamer(models)

    print("Question: What is artificial intelligence?")
    print("Responses from multiple models:")

    model_outputs = {name: "" for name in models.keys()}

    async for chunk_dict in streamer.stream_multiple_models("What is artificial intelligence?"):
        for model_name, chunk in chunk_dict.items():
            model_outputs[model_name] += chunk
            # Show partial output
            if len(chunk_dict) == 1:  # Only one model responded
                print(f"{model_name}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")

    print("\nFinal outputs:")
    for model_name, output in model_outputs.items():
        print(f"\n{model_name} (length: {len(output)}):")
        print(output[:200] + "..." if len(output) > 200 else output)


def example_buffered_streaming():
    """Example: Buffered streaming for smoother output."""
    print("\n🌊 Buffered Streaming Example")
    print("-" * 31)

    streamer = StreamingChatManager()
    buffered_streamer = BufferedStreamer(streamer, buffer_size=30, buffer_timeout=0.2)

    print("Question: Write a short poem about technology")
    print("Buffered response:")

    for buffer_chunk in buffered_streamer.stream_buffered(
        "Write a short poem about technology and human connection"
    ):
        print(f"[Buffer: {len(buffer_chunk)} chars] {buffer_chunk}")

    print()


def example_streaming_with_error_handling():
    """Example: Streaming with proper error handling."""
    print("\n🌊 Streaming with Error Handling Example")
    print("-" * 42)

    streamer = StreamingChatManager()

    async def safe_stream_with_retry(input_text: str, max_retries: int = 3):
        """Stream with retry logic on errors."""
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}")
                async for chunk in streamer.astream_response(input_text):
                    print(chunk, end="", flush=True)
                print("\n✅ Success!")
                return
            except Exception as e:
                print(f"\n❌ Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in {2 ** attempt} seconds...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    print("❌ All retry attempts failed")
                    raise

    # Test with a potentially problematic input
    await safe_stream_with_retry("Generate a response that might cause issues")


async def main():
    """Run all streaming examples."""
    print("🌊 LangChain 1.0 Streaming Examples")
    print("=" * 40)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key'")
        return

    try:
        # Run examples
        example_basic_streaming()
        await example_async_streaming()
        example_callback_streaming()
        example_analyzed_streaming()
        await example_multi_model_streaming()
        example_buffered_streaming()
        await example_streaming_with_error_handling()

        print("\n" + "=" * 40)
        print("✅ All streaming examples completed!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys configured")
        print("   - Required packages installed")
        print("   - Internet connection")


if __name__ == "__main__":
    """Run the streaming examples."""
    import os
    import sys

    # Check Python version for async support
    if sys.version_info < (3, 7):
        print("❌ This script requires Python 3.7 or higher for async/await support")
        sys.exit(1)

    # Run the async main function
    asyncio.run(main())