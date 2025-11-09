#!/usr/bin/env python3
"""
Memory Integration Templates for LangChain 1.0

This script provides templates and examples for implementing various types of
conversation memory in LangChain 1.0 applications.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.memory import BaseMemory, ConversationBufferMemory
from langchain_openai import ChatOpenAI


class ConversationMemoryManager:
    """
    Comprehensive memory management system for LangChain 1.0 conversations.

    Supports multiple memory types and strategies for different use cases.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize memory manager.

        Args:
            model_name: Model to use for memory operations
            temperature: Model temperature
        """
        self.model = ChatOpenAI(model=model_name, temperature=temperature)
        self.sessions: Dict[str, BaseChatMessageHistory] = {}

    def create_buffer_memory_chain(
        self,
        system_prompt: str,
        session_id: str = "default"
    ):
        """
        Create a chain with buffer memory (stores all messages).

        Args:
            system_prompt: System instruction for the AI
            session_id: Session identifier

        Returns:
            Chain with buffer memory
        """
        # Create prompt with message placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # Create base chain
        chain = prompt | self.model

        # Create chain with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_history

    def create_summary_memory_chain(
        self,
        system_prompt: str,
        session_id: str = "default",
        max_tokens: int = 1000
    ):
        """
        Create a chain with summary memory (summarizes old conversations).

        Args:
            system_prompt: System instruction
            session_id: Session identifier
            max_tokens: Maximum tokens for conversation before summarization

        Returns:
            Chain with summary memory
        """
        class SummaryMemory:
            def __init__(self, max_tokens: int = 1000):
                self.max_tokens = max_tokens
                self.summary = ""
                self.recent_messages = []

            def add_message(self, message: str, is_user: bool = True):
                """Add a message and manage summarization."""
                self.recent_messages.append((message, is_user, datetime.now()))

                # Simple token count approximation
                total_chars = sum(len(msg[0]) for msg in self.recent_messages)
                estimated_tokens = total_chars // 4

                if estimated_tokens > self.max_tokens:
                    self._summarize_old_messages()

            def _summarize_old_messages(self):
                """Summarize older messages to save space."""
                if len(self.recent_messages) < 2:
                    return

                # Keep recent messages, summarize older ones
                keep_count = max(2, len(self.recent_messages) // 2)
                old_messages = self.recent_messages[:-keep_count]
                self.recent_messages = self.recent_messages[-keep_count:]

                # Create summary prompt
                old_text = "\n".join([
                    f"{'User' if is_user else 'AI'}: {msg}"
                    for msg, is_user, _ in old_messages
                ])

                summary_prompt = f"""Summarize this conversation concisely:

Previous context: {self.summary}

Recent conversation:
{old_text}

Provide a brief summary that captures the key points:"""

                # This would need a model call in practice
                # For now, just concatenate
                if self.summary:
                    self.summary = f"{self.summary}\n{old_text}"
                else:
                    self.summary = old_text

            def get_context(self) -> str:
                """Get formatted context for prompting."""
                context = ""
                if self.summary:
                    context += f"Previous conversation summary: {self.summary}\n\n"

                if self.recent_messages:
                    context += "Recent messages:\n"
                    for msg, is_user, _ in self.recent_messages:
                        role = "User" if is_user else "AI"
                        context += f"{role}: {msg}\n"

                return context

        # Create prompt that accepts formatted context
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}\n\nConversation context:\n{{context}}"),
            ("human", "{input}"),
        ])

        # Create summary memory instance
        summary_memory = SummaryMemory(max_tokens)

        # Create chain with custom memory logic
        def add_memory_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Add memory context to inputs."""
            context = summary_memory.get_context()
            return {**inputs, "context": context}

        def update_memory(inputs: Dict[str, Any], response: str):
            """Update memory with new conversation."""
            summary_memory.add_message(inputs["input"], is_user=True)
            summary_memory.add_message(response, is_user=False)

        # Create chain
        chain = (
            add_memory_context
            | prompt
            | self.model
        )

        return chain, summary_memory

    def create_token_buffer_memory_chain(
        self,
        system_prompt: str,
        session_id: str = "default",
        max_tokens: int = 2000
    ):
        """
        Create a chain with token buffer memory (keeps recent messages within token limit).

        Args:
            system_prompt: System instruction
            session_id: Session identifier
            max_tokens: Maximum tokens to keep in memory

        Returns:
            Chain with token buffer memory
        """
        class TokenBufferMemory:
            def __init__(self, max_tokens: int = 2000):
                self.max_tokens = max_tokens
                self.messages = []

            def add_message(self, message: str, is_user: bool = True):
                """Add message and trim if necessary."""
                # Create message object
                if is_user:
                    msg_obj = HumanMessage(content=message)
                else:
                    msg_obj = AIMessage(content=message)

                self.messages.append(msg_obj)
                self._trim_messages()

            def _trim_messages(self):
                """Remove oldest messages to stay within token limit."""
                # Simple approximation: 4 characters per token
                total_chars = sum(len(msg.content) for msg in self.messages)
                estimated_tokens = total_chars // 4

                while estimated_tokens > self.max_tokens and len(self.messages) > 2:
                    # Remove oldest message (keep at least 2)
                    removed = self.messages.pop(0)
                    total_chars -= len(removed.content)
                    estimated_tokens = total_chars // 4

            def get_messages(self) -> List:
                """Get current messages."""
                return self.messages.copy()

        # Create prompt with message placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # Create token buffer memory
        token_memory = TokenBufferMemory(max_tokens)

        # Create chain with custom memory handling
        def prepare_memory_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Prepare input with memory messages."""
            return {
                "input": inputs["input"],
                "history": token_memory.get_messages()
            }

        def update_memory(inputs: Dict[str, Any], response: str):
            """Update memory with new conversation."""
            token_memory.add_message(inputs["input"], is_user=True)
            token_memory.add_message(response, is_user=False)

        # Create chain
        chain = (
            prepare_memory_input
            | prompt
            | self.model
        )

        return chain, token_memory

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]

    def create_multi_session_manager(self):
        """
        Create a manager that can handle multiple independent sessions.
        """
        class MultiSessionManager:
            def __init__(self, memory_manager):
                self.memory_manager = memory_manager
                self.session_chains = {}

            def get_chain(self, session_id: str, memory_type: str = "buffer"):
                """Get or create a chain for a specific session."""
                if session_id not in self.session_chains:
                    system_prompt = "You are a helpful assistant."

                    if memory_type == "buffer":
                        chain = self.memory_manager.create_buffer_memory_chain(
                            system_prompt, session_id
                        )
                    elif memory_type == "summary":
                        chain, _ = self.memory_manager.create_summary_memory_chain(
                            system_prompt, session_id
                        )
                    elif memory_type == "token":
                        chain, _ = self.memory_manager.create_token_buffer_memory_chain(
                            system_prompt, session_id
                        )
                    else:
                        raise ValueError(f"Unsupported memory type: {memory_type}")

                    self.session_chains[session_id] = chain

                return self.session_chains[session_id]

            def clear_session(self, session_id: str):
                """Clear a specific session."""
                if session_id in self.sessions:
                    self.sessions[session_id].clear()
                if session_id in self.session_chains:
                    del self.session_chains[session_id]

            def list_sessions(self) -> List[str]:
                """List all active sessions."""
                return list(self.sessions.keys())

        return MultiSessionManager(self)


def example_buffer_memory():
    """Example: Using buffer memory."""
    print("🧠 Buffer Memory Example")
    print("-" * 30)

    memory_manager = ConversationMemoryManager()
    chain = memory_manager.create_buffer_memory_chain(
        "You are a helpful assistant with good memory.",
        "buffer_example"
    )

    # Simulate conversation
    conversation = [
        "Hi, my name is John and I love hiking.",
        "What's my name?",
        "What hobbies did I mention?",
        "Can you recommend a good hiking trail near San Francisco?"
    ]

    for message in conversation:
        print(f"User: {message}")
        response = chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": "buffer_example"}}
        )
        print(f"AI: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")
        print()


def example_summary_memory():
    """Example: Using summary memory."""
    print("🧠 Summary Memory Example")
    print("-" * 30)

    memory_manager = ConversationMemoryManager()
    chain, summary_memory = memory_manager.create_summary_memory_chain(
        "You are a helpful assistant.",
        "summary_example",
        max_tokens=200  # Small limit to trigger summarization
    )

    # Long conversation to trigger summarization
    long_conversation = [
        "Tell me about artificial intelligence.",
        "What are the main types of machine learning?",
        "How do neural networks work?",
        "What's the difference between deep learning and machine learning?",
        "Explain natural language processing.",
        "What are transformers in AI?",
        "How does GPT work?",
        "What are ethical considerations in AI?"
    ]

    for i, message in enumerate(long_conversation):
        print(f"User {i+1}: {message}")
        response = chain.invoke({"input": message})
        print(f"AI {i+1}: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")

        # Show summary when it gets created
        if summary_memory.summary:
            print(f"📝 Summary: {summary_memory.summary[:150]}{'...' if len(summary_memory.summary) > 150 else ''}")
        print()


def example_token_buffer_memory():
    """Example: Using token buffer memory."""
    print("🧠 Token Buffer Memory Example")
    print("-" * 30)

    memory_manager = ConversationMemoryManager()
    chain, token_memory = memory_manager.create_token_buffer_memory_chain(
        "You are a helpful assistant.",
        "token_example",
        max_tokens=300  # Small limit to show trimming
    )

    # Add messages to exceed token limit
    for i in range(5):
        long_message = f"Message {i+1}: " + "This is a longer message to consume tokens. " * 10
        print(f"User: Message {i+1} (Length: {len(long_message)})")

        response = chain.invoke({"input": long_message})
        print(f"AI: Response {i+1}")
        print(f"Messages in memory: {len(token_memory.get_messages())}")
        print()


def example_multi_session():
    """Example: Multi-session management."""
    print("🧠 Multi-Session Example")
    print("-" * 30)

    memory_manager = ConversationMemoryManager()
    session_manager = memory_manager.create_multi_session_manager()

    # Create multiple sessions
    sessions = {
        "work": "I need help with my Python project. I'm working on a web API.",
        "personal": "I want to learn how to cook Italian food. What should I start with?",
        "study": "Can you help me understand calculus derivatives?"
    }

    for session_id, initial_message in sessions.items():
        print(f"\nSession: {session_id}")
        print(f"User: {initial_message}")

        chain = session_manager.get_chain(session_id, "buffer")
        response = chain.invoke(
            {"input": initial_message},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")

    # Continue conversation in work session
    print(f"\nContinuing work session...")
    chain = session_manager.get_chain("work", "buffer")
    response = chain.invoke(
        {"input": "What framework should I use?"},
        config={"configurable": {"session_id": "work"}}
    )
    print(f"User: What framework should I use?")
    print(f"AI: {response.content[:100]}{'...' if len(response.content) > 100 else ''}")

    print(f"\nActive sessions: {session_manager.list_sessions()}")


if __name__ == "__main__":
    """Run memory integration examples."""
    print("🧠 LangChain 1.0 Memory Integration Examples")
    print("=" * 50)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key'")
        exit(1)

    try:
        # Run examples
        example_buffer_memory()
        example_summary_memory()
        example_token_buffer_memory()
        example_multi_session()

        print("\n" + "=" * 50)
        print("✅ All memory examples completed!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure you have:")
        print("   - Valid API keys configured")
        print("   - Required packages installed")
        print("   - Internet connection")