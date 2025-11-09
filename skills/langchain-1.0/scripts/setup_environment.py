#!/usr/bin/env python3
"""
LangChain 1.0 Environment Setup Script

This script sets up a complete LangChain 1.0 development environment
with all necessary dependencies and configuration.
"""

import subprocess
import sys
from pathlib import Path


def install_package(package: str, description: str = "") -> bool:
    """Install a package using pip with error handling."""
    try:
        print(f"Installing {package}{' - ' + description if description else ''}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    """Set up LangChain 1.0 environment."""
    print("🚀 Setting up LangChain 1.0 development environment...")

    # Core LangChain packages
    core_packages = [
        ("langchain>=1.0.0", "Main LangChain library"),
        ("langchain-core>=0.1.0", "Core LangChain components"),
        ("langchain-community>=0.0.20", "Community integrations"),
    ]

    # Model provider packages
    model_packages = [
        ("langchain-openai>=0.1.0", "OpenAI integration"),
        ("langchain-anthropic>=0.1.0", "Anthropic integration"),
        ("langchain-google-genai>=1.0.0", "Google AI integration"),
    ]

    # Document processing
    document_packages = [
        ("langchain-text-splitters>=0.0.1", "Text splitting utilities"),
        ("langchain-chroma>=0.1.0", "Chroma vector store"),
        ("pypdf>=3.17.0", "PDF processing"),
        ("python-docx>=1.1.0", "Word document processing"),
        ("beautifulsoup4>=4.12.0", "HTML parsing"),
    ]

    # Additional utilities
    utility_packages = [
        ("tiktoken>=0.5.0", "Token counting"),
        ("faiss-cpu>=1.7.0", "FAISS vector store"),
        ("pydantic>=2.0.0", "Data validation"),
        ("python-dotenv>=1.0.0", "Environment variables"),
    ]

    # Install package groups
    package_groups = [
        ("Core LangChain", core_packages),
        ("Model Providers", model_packages),
        ("Document Processing", document_packages),
        ("Utilities", utility_packages),
    ]

    failed_packages = []

    for group_name, packages in package_groups:
        print(f"\n📦 Installing {group_name} packages...")
        for package, description in packages:
            if not install_package(package, description):
                failed_packages.append(package)

    # Create .env template
    env_template = """# LangChain 1.0 Environment Variables
# Copy this file to .env and fill in your API keys

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Other model providers
HUGGINGFACEHUB_API_KEY=your_huggingface_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Optional: Vector stores
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
"""

    env_path = Path(".env.template")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_template)
        print(f"\n✅ Created environment template: {env_path}")
        print("📝 Copy to .env and add your API keys")

    # Summary
    print("\n" + "="*50)
    print("🎉 LangChain 1.0 setup complete!")

    if failed_packages:
        print(f"\n⚠️  Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"   - {package}")
        print("\n💡 Install them manually with: pip install <package>")

    print("\n📚 Next steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Test basic functionality with: python -c 'import langchain; print(langchain.__version__)'")
    print("3. Check out the templates in assets/templates/")
    print("4. Read references/model_providers.md for configuration details")


if __name__ == "__main__":
    main()