#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Setup Script for LangChain 1.0

This script provides utilities and templates for setting up RAG systems
with various vector stores and document processors.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)

# Vector stores
try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Models
from langchain_openai import ChatOpenAI


class RAGSystem:
    """
    Complete RAG system with document loading, processing, and retrieval.
    """

    def __init__(
        self,
        vector_store_type: str = "chroma",
        embedding_model: str = "openai",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize RAG system.

        Args:
            vector_store_type: Type of vector store ('chroma', 'faiss')
            embedding_model: Embedding model to use
            persist_directory: Directory for persisting vector store
        """
        self.vector_store_type = vector_store_type.lower()
        self.embedding_model = embedding_model.lower()
        self.persist_directory = persist_directory

        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()

        # Initialize vector store
        self.vectorstore = None

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        if self.embedding_model == "openai":
            return OpenAIEmbeddings()
        elif self.embedding_model == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def load_documents(
        self,
        source_path: str,
        file_type: str = "auto",
        recursive: bool = True
    ) -> List[Document]:
        """
        Load documents from various sources.

        Args:
            source_path: Path to file or directory
            file_type: Type of files to load ('auto', 'pdf', 'txt', 'docx', 'csv')
            recursive: Whether to load recursively from directories

        Returns:
            List of loaded documents
        """
        source_path = Path(source_path)

        if source_path.is_file():
            # Load single file
            return self._load_single_file(source_path)
        elif source_path.is_dir():
            # Load directory
            return self._load_directory(source_path, file_type, recursive)
        else:
            raise FileNotFoundError(f"Source path not found: {source_path}")

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension."""
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
        elif extension == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif extension == ".csv":
            loader = CSVLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        return loader.load()

    def _load_directory(
        self,
        directory: Path,
        file_type: str,
        recursive: bool
    ) -> List[Document]:
        """Load all files from a directory."""
        if file_type == "auto":
            # Load all supported file types
            glob_pattern = "**/*" if recursive else "*"
            loader = DirectoryLoader(
                str(directory),
                glob=glob_pattern,
                recursive=recursive,
                show_progress=True
            )
        else:
            # Load specific file type
            patterns = {
                "pdf": "**/*.pdf",
                "txt": "**/*.txt",
                "md": "**/*.md",
                "docx": "**/*.docx",
                "csv": "**/*.csv"
            }

            if file_type not in patterns:
                raise ValueError(f"Unsupported file type: {file_type}")

            glob_pattern = patterns[file_type]
            if not recursive:
                glob_pattern = glob_pattern.replace("**/", "")

            loader = DirectoryLoader(
                str(directory),
                glob=glob_pattern,
                recursive=recursive,
                show_progress=True
            )

        return loader.load()

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive"
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            splitter_type: Type of splitter ('recursive', 'character')

        Returns:
            List of split documents
        """
        if splitter_type == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        elif splitter_type == "character":
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        else:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")

        return splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]):
        """
        Create and populate vector store.

        Args:
            documents: Documents to add to vector store
        """
        if self.vector_store_type == "chroma" and CHROMA_AVAILABLE:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        elif self.vector_store_type == "faiss" and FAISS_AVAILABLE:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        else:
            raise ValueError(f"Vector store {self.vector_store_type} not available")

    def load_vector_store(self):
        """Load existing vector store from disk."""
        if self.vector_store_type == "chroma" and CHROMA_AVAILABLE:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
        return False

    def create_retrieval_chain(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        retrieval_k: int = 4
    ):
        """
        Create a retrieval-augmented generation chain.

        Args:
            model_name: Model to use for generation
            temperature: Model temperature
            retrieval_k: Number of documents to retrieve

        Returns:
            RAG chain ready for invocation
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieval_k}
        )

        # Create prompt template
        template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, say so clearly."""

        prompt = ChatPromptTemplate.from_template(template)

        # Initialize model
        model = ChatOpenAI(model=model_name, temperature=temperature)

        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        return rag_chain

    def add_documents(self, documents: List[Document]):
        """Add documents to existing vector store."""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
        else:
            raise ValueError("Vector store not initialized")

    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        if self.vectorstore:
            return self.vectorstore.similarity_search(query, k=k)
        else:
            raise ValueError("Vector store not initialized")


def setup_rag_from_directory(
    directory_path: str,
    vector_store_type: str = "chroma",
    persist_directory: str = "./chroma_db"
) -> RAGSystem:
    """
    Complete RAG setup from a directory of documents.

    Args:
        directory_path: Path to directory containing documents
        vector_store_type: Type of vector store
        persist_directory: Directory to persist vector store

    Returns:
        Configured RAG system
    """
    # Initialize RAG system
    rag = RAGSystem(
        vector_store_type=vector_store_type,
        persist_directory=persist_directory
    )

    # Load documents
    print("📄 Loading documents...")
    documents = rag.load_documents(directory_path)
    print(f"✅ Loaded {len(documents)} documents")

    # Split documents
    print("✂️  Splitting documents...")
    split_docs = rag.split_documents(documents)
    print(f"✅ Created {len(split_docs)} chunks")

    # Create vector store
    print("🗄️  Creating vector store...")
    rag.create_vector_store(split_docs)
    print("✅ Vector store created successfully")

    return rag


if __name__ == "__main__":
    print("🔍 LangChain 1.0 RAG Setup Examples")
    print("=" * 40)

    # Example usage (requires documents directory and API keys)
    try:
        # Setup RAG from documents directory
        rag = setup_rag_from_directory(
            directory_path="./documents",
            vector_store_type="chroma"
        )

        # Create retrieval chain
        chain = rag.create_retrieval_chain()

        # Test the chain
        question = "What are the main topics covered in these documents?"
        answer = chain.invoke(question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("   - API keys configured in .env")
        print("   - A documents directory with supported files")
        print("   - Required packages installed")