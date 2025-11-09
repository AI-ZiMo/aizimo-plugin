#!/usr/bin/env python3
"""
Simple Chatbot Example for LangChain 1.0

This example demonstrates a complete chatbot implementation with conversation memory,
proper error handling, and streaming responses using LangChain 1.0.
"""

import os
import sys
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


class SimpleChatbot:
    """
    A simple yet feature-rich chatbot using LangChain 1.0.

    Features:
    - Conversation memory
    - Streaming responses
    - Error handling
    - Customizable personality
    - Session management
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful, friendly, and knowledgeable assistant.",
        max_history_length: int = 10
    ):
        """
        Initialize the chatbot.

        Args:
            model_name: OpenAI model to use
            temperature: Model temperature (0.0-1.0)
            system_prompt: System instruction for the AI
            max_history_length: Maximum number of messages to keep in memory
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_history_length = max_history_length

        # Initialize model
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True
        )

        # Create prompt template with memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # Create base chain
        self.chain = self.prompt | self.model | StrOutputParser()

        # Set up chat history
        self.chat_history = InMemoryChatMessageHistory()
        self.sessions: Dict[str, BaseChatMessageHistory] = {}

        # Create chain with message history
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
        return self.sessions[session_id]

    def chat(self, message: str, session_id: str = "default") -> str:
        """
        Send a message and get a response.

        Args:
            message: User's message
            session_id: Session identifier for conversation memory

        Returns:
            AI's response
        """
        try:
            response = self.chain_with_history.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def chat_stream(self, message: str, session_id: str = "default"):
        """
        Send a message and stream the response.

        Args:
            message: User's message
            session_id: Session identifier for conversation memory

        Yields:
            Response chunks as they arrive
        """
        try:
            for chunk in self.chain_with_history.stream(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            ):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def get_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        if session_id not in self.sessions:
            return []

        history = []
        for message in self.sessions[session_id].messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})

        return history

    def set_personality(self, new_system_prompt: str):
        """Update the chatbot's personality."""
        self.system_prompt = new_system_prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", new_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.model | StrOutputParser()
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="history",
        )


def interactive_chat():
    """Run an interactive chat session."""
    print("🤖 LangChain 1.0 Simple Chatbot")
    print("=" * 40)
    print("Type 'quit' to exit, 'clear' to clear history, 'help' for commands")
    print("-" * 40)

    # Initialize chatbot
    chatbot = SimpleChatbot(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        system_prompt="You are a helpful, friendly, and knowledgeable assistant. Be conversational and engaging."
    )

    session_id = "interactive_session"

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history(session_id)
                print("🧹 Conversation history cleared.")
                continue
            elif user_input.lower() == 'help':
                print("\n📚 Available commands:")
                print("  quit   - Exit the chatbot")
                print("  clear  - Clear conversation history")
                print("  help   - Show this help message")
                print("  personality <name> - Change personality (expert, casual, poetic)")
                continue
            elif user_input.lower().startswith('personality'):
                parts = user_input.split(' ', 1)
                if len(parts) > 1:
                    personality = parts[1].lower()
                    personalities = {
                        'expert': "You are an expert in your field. Provide detailed, accurate, and well-structured answers.",
                        'casual': "You are a friendly, casual conversation partner. Be relaxed and use informal language.",
                        'poetic': "You are a poetic and creative soul. Express yourself with beautiful language and metaphors.",
                        'professional': "You are a professional assistant. Be formal, precise, and business-oriented."
                    }
                    if personality in personalities:
                        chatbot.set_personality(personalities[personality])
                        print(f"✨ Personality changed to: {personality}")
                    else:
                        print(f"❌ Unknown personality. Available: {', '.join(personalities.keys())}")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_history(session_id)
                if history:
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(history[-5:], 1):  # Show last 5 messages
                        role = "You" if msg['role'] == 'user' else "AI"
                        print(f"  {i}. {role}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
                else:
                    print("📜 No conversation history yet.")
                continue

            if not user_input:
                continue

            # Get streaming response
            print("🤖 AI: ", end="", flush=True)
            response_chunks = []
            for chunk in chatbot.chat_stream(user_input, session_id):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print()  # New line after complete response

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def demo_conversations():
    """Demonstrate different conversation scenarios."""
    print("\n🎭 Demo: Different Conversation Scenarios")
    print("=" * 50)

    # Scenario 1: Technical Assistant
    print("\n1️⃣  Technical Assistant Demo")
    print("-" * 30)

    tech_chatbot = SimpleChatbot(
        system_prompt="You are a technical support specialist. Provide clear, step-by-step solutions to technical problems.",
        temperature=0.2
    )

    tech_questions = [
        "How do I fix a slow computer?",
        "What's the difference between RAM and ROM?",
        "How can I improve my Wi-Fi signal?"
    ]

    for question in tech_questions:
        print(f"\nUser: {question}")
        response = tech_chatbot.chat(question, "tech_session")
        print(f"AI: {response[:200]}{'...' if len(response) > 200 else ''}")

    # Scenario 2: Creative Writing Assistant
    print("\n\n2️⃣  Creative Writing Assistant Demo")
    print("-" * 30)

    creative_chatbot = SimpleChatbot(
        system_prompt="You are a creative writing assistant. Help users with storytelling, poetry, and creative expression.",
        temperature=0.8
    )

    creative_prompts = [
        "Help me start a sci-fi story",
        "Give me a metaphor for loneliness",
        "Suggest a plot twist for a mystery novel"
    ]

    for prompt in creative_prompts:
        print(f"\nUser: {prompt}")
        response = creative_chatbot.chat(prompt, "creative_session")
        print(f"AI: {response[:200]}{'...' if len(response) > 200 else ''}")

    # Scenario 3: Learning Tutor
    print("\n\n3️⃣  Learning Tutor Demo")
    print("-" * 30)

    tutor_chatbot = SimpleChatbot(
        system_prompt="You are a patient and encouraging tutor. Explain concepts clearly and ask follow-up questions to ensure understanding.",
        temperature=0.5
    )

    learning_questions = [
        "Can you explain photosynthesis simply?",
        "Why is the sky blue?",
        "How does electricity work?"
    ]

    for question in learning_questions:
        print(f"\nUser: {question}")
        response = tutor_chatbot.chat(question, "tutor_session")
        print(f"AI: {response[:200]}{'...' if len(response) > 200 else ''}")


def test_chatbot_features():
    """Test various chatbot features."""
    print("\n🧪 Testing Chatbot Features")
    print("=" * 30)

    chatbot = SimpleChatbot()
    session_id = "test_session"

    # Test basic conversation
    print("\n1. Testing basic conversation...")
    response = chatbot.chat("Hello! Can you help me?", session_id)
    print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

    # Test conversation memory
    print("\n2. Testing conversation memory...")
    chatbot.chat("My name is Alice", session_id)
    response = chatbot.chat("What's my name?", session_id)
    print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")

    # Test history retrieval
    print("\n3. Testing history retrieval...")
    history = chatbot.get_history(session_id)
    print(f"Number of messages in history: {len(history)}")

    # Test history clearing
    print("\n4. Testing history clearing...")
    chatbot.clear_history(session_id)
    history_after_clear = chatbot.get_history(session_id)
    print(f"Messages after clearing: {len(history_after_clear)}")

    # Test streaming
    print("\n5. Testing streaming...")
    print("Streaming response: ", end="", flush=True)
    for chunk in chatbot.chat_stream("Count to 5", "stream_test"):
        print(chunk, end="", flush=True)
    print()

    print("\n✅ Feature testing completed!")


if __name__ == "__main__":
    """Main entry point with multiple demo options."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "interactive":
            interactive_chat()
        elif mode == "demo":
            demo_conversations()
        elif mode == "test":
            test_chatbot_features()
        else:
            print("Usage:")
            print("  python simple_chatbot.py interactive  - Interactive chat mode")
            print("  python simple_chatbot.py demo        - Demo different scenarios")
            print("  python simple_chatbot.py test        - Test features")
    else:
        # Default: run interactive chat
        interactive_chat()