# LangChain 1.0 Migration Guide

This guide covers the key changes and migration steps when moving from LangChain 0.x to 1.0.

## Major Changes

### 1. Package Structure

**Before (0.x):**
```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
```

**After (1.0):**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
```

### 2. LangChain Expression Language (LCEL)

**Old Chain Style:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

prompt = PromptTemplate(template="Tell me a joke about {topic}")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="programming")
```

**New LCEL Style:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI()
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "programming"})
```

### 3. Model Integration

**Old Style:**
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI(model="text-davinci-003")
chat_model = ChatOpenAI(model="gpt-3.5-turbo")
```

**New Style:**
```python
from langchain_openai import OpenAI, ChatOpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
chat_model = ChatOpenAI(model="gpt-3.5-turbo")
```

### 4. Memory Implementation

**Old Style:**
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

**New Style:**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.memory import BufferMemory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

memory = BufferMemory(return_messages=True)
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# Uninstall old packages
pip uninstall langchain

# Install new packages
pip install langchain>=1.0.0
pip install langchain-core
pip install langchain-community
pip install langchain-openai
pip install langchain-anthropic
```

### Step 2: Update Imports

Create a mapping of old imports to new imports:

| Old Import | New Import |
|------------|------------|
| `from langchain.chains import LLMChain` | Use LCEL syntax instead |
| `from langchain.llms import OpenAI` | `from langchain_openai import OpenAI` |
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` |
| `from langchain.prompts import PromptTemplate` | `from langchain_core.prompts import PromptTemplate` |
| `from langchain.document_loaders import PyPDFLoader` | `from langchain_community.document_loaders import PyPDFLoader` |
| `from langchain.text_splitter import RecursiveCharacterTextSplitter` | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| `from langchain.vectorstores import Chroma` | `from langchain_chroma import Chroma` |
| `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` |
| `from langchain.memory import ConversationBufferMemory` | `from langchain_core.memory import ConversationBufferMemory` |

### Step 3: Convert Chains to LCEL

**Example 1: Simple Chain**
```python
# Old
prompt = PromptTemplate(template="Hello {name}")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

# New
prompt = PromptTemplate.from_template("Hello {name}")
llm = OpenAI()
chain = prompt | llm | StrOutputParser()
```

**Example 2: Sequential Chain**
```python
# Old
from langchain.chains import SequentialChain

chain1 = LLMChain(llm=llm1, prompt=prompt1)
chain2 = LLMChain(llm=llm2, prompt=prompt2)
sequential_chain = SequentialChain(chains=[chain1, chain2])

# New
chain1 = prompt1 | llm1 | StrOutputParser()
chain2 = prompt2 | llm2 | StrOutputParser()
sequential_chain = {"input": chain1} | RunnablePassthrough.assign(output=chain2)
```

### Step 4: Update Document Processing

**Text Splitters:**
```python
# Old
from langchain.text_splitter import RecursiveCharacterTextSplitter

# New
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

**Document Loaders:**
```python
# Old
from langchain.document_loaders import PyPDFLoader

# New
from langchain_community.document_loaders import PyPDFLoader
```

### Step 5: Update Vector Stores

**Chroma:**
```python
# Old
from langchain.vectorstores import Chroma

# New
from langchain_chroma import Chroma
```

**FAISS:**
```python
# Old
from langchain.vectorstores import FAISS

# New
from langchain_community.vectorstores import FAISS
```

### Step 6: Update Memory Implementation

Memory in 1.0 requires more explicit configuration:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory

# Create prompt with message placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Configure memory
memory = ConversationBufferMemory(
    chat_memory=BaseChatMessageHistory(),
    return_messages=True
)

# Use in chain
chain = prompt | model | output_parser
```

## Breaking Changes

### 1. Chain.run() is Deprecated

**Old:**
```python
result = chain.run(input_text="Hello")
```

**New:**
```python
result = chain.invoke({"input": "Hello"})
```

### 2. Output Parsers

**Old:**
```python
from langchain.output_parsers import StrOutputParser
```

**New:**
```python
from langchain_core.output_parsers import StrOutputParser
```

### 3. Callbacks

**Old:**
```python
from langchain.callbacks import get_openai_callback
```

**New:**
```python
from langchain_core.callbacks import get_openai_callback
```

### 4. Agents

**Old:**
```python
from langchain.agents import initialize_agent, AgentType
```

**New:**
```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
```

## Common Migration Issues

### Issue 1: Import Errors

**Problem:** `ImportError: cannot import name 'LLMChain' from 'langchain.chains'`

**Solution:** LLMChain is deprecated. Use LCEL syntax instead.

### Issue 2: Memory Not Working

**Problem:** Memory not persisting between invocations

**Solution:** Use the new memory pattern with explicit message handling.

### Issue 3: Vector Store Integration

**Problem:** Vector store imports not working

**Solution:** Install the specific integration package (e.g., `langchain-chroma`).

### Issue 4: Streaming Not Working

**Problem:** Streaming functionality broken

**Solution:** Use the new streaming pattern:
```python
async for chunk in chain.astream({"input": query}):
    print(chunk.content, end="")
```

## Testing Your Migration

### Basic Test

```python
import langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def test_basic_functionality():
    """Test basic LangChain 1.0 functionality."""
    try:
        # Create a simple chain
        prompt = ChatPromptTemplate.from_template("Hello {name}")
        llm = ChatOpenAI()
        chain = prompt | llm | StrOutputParser()

        # Test invocation
        result = chain.invoke({"name": "World"})

        print(f"✅ Basic test passed: {result}")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"LangChain version: {langchain.__version__}")
    test_basic_functionality()
```

### Advanced Test

```python
def test_rag_functionality():
    """Test RAG functionality."""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        # Create test documents
        docs = [Document(page_content="Test document content")]

        # Split documents
        splitter = RecursiveCharacterTextSplitter()
        split_docs = splitter.split_documents(docs)

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(split_docs, embeddings)

        # Test retrieval
        results = vectorstore.similarity_search("test", k=1)

        print(f"✅ RAG test passed: Found {len(results)} documents")
        return True

    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False
```

## Best Practices for 1.0

1. **Use LCEL syntax** for all new chains
2. **Implement proper error handling** with try-catch blocks
3. **Use streaming** for long-running operations
4. **Configure async patterns** for better performance
5. **Use the new memory pattern** with explicit message handling
6. **Install specific integration packages** for each provider
7. **Use type hints** for better code clarity
8. **Implement proper logging** for debugging

## Getting Help

- Check the [LangChain 1.0 documentation](https://python.langchain.com/docs/get_started/introduction)
- Review the [migration guide](https://python.langchain.com/docs/versions/migrating/)
- Join the [LangChain Discord community](https://discord.gg/langchain)
- Check GitHub issues for known problems