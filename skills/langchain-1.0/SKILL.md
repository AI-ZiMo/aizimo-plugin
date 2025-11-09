---
name: langchain-1.0
description: This skill should be used when users need help with LangChain 1.0 development, including building chains, implementing RAG systems, integrating models, creating agents, or troubleshooting LangChain applications. Use this for any LangChain 1.0 specific tasks like migrating from 0.x, setting up environments, or implementing advanced patterns.
---

# LangChain 1.0 Development Skill

## Overview

This skill enables comprehensive development with LangChain 1.0, providing patterns, templates, and best practices for building chains, RAG systems, agents, and other LangChain applications. It includes migration guidance, component patterns, and optimization techniques for the 1.0 architecture.

## Core Capabilities

### 1. Environment Setup and Migration

**When to use:** Setting up new LangChain 1.0 projects or migrating from 0.x versions

**Implementation steps:**
1. Install LangChain 1.0 and core dependencies using `scripts/setup_environment.py`
2. Configure model providers following `references/model_providers.md`
3. Update existing code using `references/langchain_1_0_migration.md`
4. Test basic functionality with `templates/basic_chain_template.py`

**Common patterns:**
- Use the new LCEL (LangChain Expression Language) syntax
- Implement proper error handling and streaming
- Configure async/await patterns for better performance

### 2. Building Basic Chains

**When to use:** Creating simple LLM-powered workflows with prompt templates and model integration

**Implementation steps:**
1. Define prompt templates using `langchain_core.prompts`
2. Configure models with appropriate providers
3. Create chains using the pipe operator (`|`)
4. Add output parsers for structured responses
5. Implement streaming if needed

**Code pattern:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Create chain using LCEL
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI()
chain = prompt | llm | StrOutputParser()

# Invoke the chain
result = chain.invoke({"topic": "programming"})
```

**Templates available:** `assets/templates/basic_chain_template.py`

### 3. RAG System Implementation

**When to use:** Building retrieval-augmented generation systems for document Q&A or knowledge bases

**Implementation steps:**
1. Set up vector store using `scripts/rag_setup.py`
2. Configure document loaders and text splitters
3. Create embedding models for vectorization
4. Build retrieval chains with proper prompting
5. Implement relevance scoring and filtering

**Key components:**
- Document loaders for various file types
- Text splitters for chunking strategies
- Vector stores (Chroma, FAISS, Pinecone)
- Retrieval chains with reranking options

**Templates available:** `assets/templates/rag_template.py`, `assets/examples/document_qa.py`

### 4. Memory and Conversation Management

**When to use:** Adding conversation history or persistent memory to chat applications

**Implementation steps:**
1. Choose appropriate memory type (buffer, summary, token buffer)
2. Configure memory with LLM integration for summarization
3. Add memory to chains and agents
4. Implement context window management
5. Handle memory persistence across sessions

**Memory patterns:**
- `ConversationBufferMemory` for simple history
- `ConversationSummaryMemory` for long conversations
- `ConversationTokenBufferMemory` for token-limited contexts

**Scripts available:** `scripts/memory_integration.py`

### 5. Tool Integration and Agent Development

**When to use:** Creating agents that can use external tools, APIs, or perform multi-step reasoning

**Implementation steps:**
1. Define custom tools using the `@tool` decorator
2. Configure agent with appropriate reasoning type
3. Set up tool execution with proper error handling
4. Implement agent execution loops
5. Add monitoring and logging

**Tool creation pattern:**
```python
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for relevant information.

    Args:
        query: The search query string.
    """
    # Implementation here
    return results
```

**Examples available:** `assets/examples/tool_agent.py`

### 6. Advanced Output Parsing

**When to use:** Structuring LLM responses into specific formats like JSON, Pydantic models, or custom types

**Implementation steps:**
1. Choose appropriate parser type (JSON, Pydantic, custom)
2. Design output schemas with validation
3. Create parsing instructions in prompts
4. Handle parsing errors gracefully
5. Implement fallback mechanisms

**Parser types:**
- `JsonOutputParser` for JSON responses
- `PydanticOutputParser` for typed objects
- `CommaSeparatedListOutputParser` for lists
- Custom parsers for complex structures

**Scripts available:** `scripts/output_parser_templates.py`

### 7. Streaming and Async Implementation

**When to use:** Building real-time applications or processing multiple requests concurrently

**Implementation steps:**
1. Configure streaming for models and chains
2. Implement async chain execution
3. Handle streaming chunks properly
4. Add error handling for async operations
5. Optimize for performance and resource usage

**Streaming pattern:**
```python
async def stream_response(query: str):
    async for chunk in chain.astream({"input": query}):
        print(chunk.content, end="", flush=True)
```

**Scripts available:** `scripts/streaming_handler.py`

### 8. Message Handling and Response Analysis

**When to use:** Working with AIMessage objects, analyzing model responses, tracking token usage, or handling provider-specific response formats

**Implementation steps:**
1. Extract metadata from AIMessage objects using `scripts/message_handling_examples.py`
2. Analyze token usage patterns and calculate costs
3. Handle provider-specific response formats
4. Implement response quality analysis
5. Track conversation history and statistics

**Key concepts:**
- AIMessage structure and metadata extraction
- Token usage analysis across providers
- Cost calculation and monitoring
- Response quality metrics
- Conversation tracking and serialization

**Code pattern:**
```python
from langchain_core.messages import AIMessage

# Extract comprehensive information from AI responses
def analyze_response(message: AIMessage) -> dict:
    return {
        "content": message.content,
        "token_usage": message.usage_metadata,
        "model_info": message.response_metadata,
        "cost": calculate_cost(message)
    }

# Handle different provider formats
token_usage = message.usage_metadata or message.response_metadata.get("token_usage", {})
```

**References available:** `references/message_types_and_responses.md`
**Scripts available:** `scripts/message_handling_examples.py`

## Decision Tree

**Start here to determine the right approach:**

1. **New to LangChain 1.0?** → Use Environment Setup and Migration
2. **Need simple LLM interaction?** → Build Basic Chains
3. **Working with documents/knowledge?** → Implement RAG System
4. **Building a chatbot?** → Add Memory and Conversation Management
5. **Need external integrations?** → Create Tools and Agents
6. **Need structured outputs?** → Use Advanced Output Parsing
7. **Real-time requirements?** → Implement Streaming and Async
8. **Working with model responses?** → Use Message Handling and Response Analysis

## Common Troubleshooting

**Migration Issues:**
- Check `references/langchain_1_0_migration.md` for breaking changes
- Update import statements: `from langchain.chains` → `from langchain_core`
- Replace deprecated methods with LCEL syntax

**Performance Optimization:**
- Use streaming for long responses
- Implement caching for repeated queries
- Batch process when possible
- Monitor token usage and costs

**Error Handling:**
- Always wrap chain invocations in try-catch blocks
- Implement retry logic for API failures
- Use proper logging for debugging
- Validate inputs before processing

## Resources

This skill includes comprehensive resource directories organized by type:

### scripts/
Executable code for common LangChain operations and setup tasks.

**Available scripts:**
- `setup_environment.py` - Environment configuration and dependency installation
- `create_basic_chain.py` - Template for creating basic LLM chains
- `rag_setup.py` - Vector store and RAG system setup
- `memory_integration.py` - Memory implementation templates
- `output_parser_templates.py` - Common output parser implementations
- `streaming_handler.py` - Streaming response handler template
- `message_handling_examples.py` - Comprehensive AIMessage handling and analysis examples

**Usage:** Execute scripts directly for automation or read them for code patterns and customization.

### references/
Documentation and guidance materials for informed development decisions.

**Available references:**
- `langchain_1_0_migration.md` - Complete migration guide from 0.x to 1.0
- `component_patterns.md` - Design patterns for LangChain components
- `model_providers.md` - Configuration guides for different LLM providers
- `message_types_and_responses.md` - Comprehensive guide to AIMessage objects and response handling
- `prompt_templates.md` - Collection of effective prompt templates
- `troubleshooting.md` - Common issues and their solutions
- `best_practices.md` - Performance optimization and architectural guidance

**Usage:** Load into context when needing detailed reference information or guidance.

### assets/
Templates, examples, and boilerplate code for rapid development.

**Available assets:**
- `templates/` - Reusable code templates
  - `basic_chain_template.py` - Starting point for simple chains
  - `rag_template.py` - RAG system implementation template
  - `conversation_template.py` - Chatbot with memory template
- `examples/` - Complete working implementations
  - `simple_chatbot.py` - Basic conversational AI
  - `document_qa.py` - Document question-answering system
  - `tool_agent.py` - Agent with external tools integration

**Usage:** Copy and modify templates, or study examples for implementation patterns.

## Resources

This skill includes example resource directories that demonstrate how to organize different types of bundled resources:

### scripts/
Executable code (Python/Bash/etc.) that can be run directly to perform specific operations.

**Examples from other skills:**
- PDF skill: `fill_fillable_fields.py`, `extract_form_field_info.py` - utilities for PDF manipulation
- DOCX skill: `document.py`, `utilities.py` - Python modules for document processing

**Appropriate for:** Python scripts, shell scripts, or any executable code that performs automation, data processing, or specific operations.

**Note:** Scripts may be executed without loading into context, but can still be read by Claude for patching or environment adjustments.

### references/
Documentation and reference material intended to be loaded into context to inform Claude's process and thinking.

**Examples from other skills:**
- Product management: `communication.md`, `context_building.md` - detailed workflow guides
- BigQuery: API reference documentation and query examples
- Finance: Schema documentation, company policies

**Appropriate for:** In-depth documentation, API references, database schemas, comprehensive guides, or any detailed information that Claude should reference while working.

### assets/
Files not intended to be loaded into context, but rather used within the output Claude produces.

**Examples from other skills:**
- Brand styling: PowerPoint template files (.pptx), logo files
- Frontend builder: HTML/React boilerplate project directories
- Typography: Font files (.ttf, .woff2)

**Appropriate for:** Templates, boilerplate code, document templates, images, icons, fonts, or any files meant to be copied or used in the final output.

---

**Any unneeded directories can be deleted.** Not every skill requires all three types of resources.
