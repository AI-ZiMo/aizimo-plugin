# Aizimo Plugin

A comprehensive Claude plugin for AI development skills, featuring LangChain 1.0 expertise and best practices.

## Overview

The Aizimo Plugin provides a collection of agent skills designed to enhance Claude's capabilities in various AI development domains. Currently featuring comprehensive LangChain 1.0 development support, this plugin enables sophisticated AI application building, RAG system implementation, and advanced chain development.

## Features

### 🚀 LangChain 1.0 Development Skill

Complete support for LangChain 1.0 development with:

- **Environment Setup & Migration**: Seamless setup for new projects and migration from LangChain 0.x to 1.0
- **Chain Building**: Create powerful LLM-powered workflows using LCEL (LangChain Expression Language)
- **RAG Systems**: Build retrieval-augmented generation systems for document Q&A and knowledge bases
- **Memory Management**: Implement conversation history and persistent memory in chat applications
- **Tool Integration**: Create agents with external tool integration and multi-step reasoning
- **Output Parsing**: Structure LLM responses into JSON, Pydantic models, or custom formats
- **Streaming & Async**: Real-time applications with async processing capabilities
- **Message Handling**: Comprehensive response analysis, token tracking, and cost monitoring

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/aizimo-plugin.git
cd aizimo-plugin
```

2. Install as a Claude plugin by placing it in your Claude plugins directory:
```bash
~/.claude/plugins/marketplaces/aizimo-plugin
```

3. The plugin will be automatically loaded by Claude upon restart.

## Plugin Structure

```
aizimo-plugin/
├── .claude-plugin/
│   └── marketplace.json         # Plugin configuration
└── skills/
    └── langchain-1.0/
        ├── SKILL.md             # Skill documentation
        ├── scripts/             # Executable scripts
        │   ├── setup_environment.py
        │   ├── create_basic_chain.py
        │   ├── rag_setup.py
        │   ├── memory_integration.py
        │   ├── output_parser_templates.py
        │   ├── streaming_handler.py
        │   └── message_handling_examples.py
        ├── references/          # Reference documentation
        │   ├── langchain_1_0_migration.md
        │   ├── message_types_and_responses.md
        │   └── model_providers.md
        └── assets/              # Templates and examples
            ├── templates/
            │   └── basic_chain_template.py
            └── examples/
                └── simple_chatbot.py
```

## Usage

Once installed, Claude will automatically have access to the skills provided by this plugin. The LangChain 1.0 skill is activated when:

- Users need help with LangChain 1.0 development
- Building chains, RAG systems, or agents
- Implementing LangChain applications
- Migrating from LangChain 0.x
- Troubleshooting LangChain applications

### Example Use Cases

1. **Setting up a new LangChain project**:
   - Automatic environment configuration
   - Model provider setup
   - Basic chain templates

2. **Building a RAG system**:
   - Vector store configuration
   - Document loading and chunking
   - Retrieval chain implementation

3. **Creating conversational AI**:
   - Memory integration
   - Conversation management
   - Context window handling

4. **Developing agents with tools**:
   - Custom tool creation
   - Agent configuration
   - Error handling and monitoring

## LangChain 1.0 Resources

### Scripts
Executable Python scripts for common operations:
- Environment setup and dependency management
- Chain creation templates
- RAG system configuration
- Memory integration patterns
- Output parser examples
- Streaming response handlers
- Message handling and analysis

### References
Detailed documentation for informed decisions:
- Migration guide from LangChain 0.x to 1.0
- Model provider configuration guides
- Message types and response handling
- Best practices and optimization tips

### Assets
Ready-to-use templates and examples:
- Basic chain templates
- RAG system implementations
- Conversation templates
- Working chatbot examples

## Configuration

The plugin is configured via `.claude-plugin/marketplace.json`:

```json
{
  "name": "aizimo-plugin",
  "owner": {
    "name": "AI Zimo",
    "email": "2876253980@qq.com"
  },
  "metadata": {
    "description": "Aizimo plugin",
    "version": "1.0.0"
  },
  "plugins": [
    {
      "name": "Agent-skills",
      "description": "Collection of agent skills, Such as LangChain 1.0",
      "source": "./",
      "strict": false,
      "skills": [
        "./skills/langchain-1.0"
      ]
    }
  ]
}
```

## Contributing

Contributions are welcome! To add new skills or improve existing ones:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-skill`)
3. Add your skill following the existing structure
4. Update documentation
5. Submit a pull request

### Adding New Skills

To add a new skill to the plugin:

1. Create a new directory under `skills/`
2. Add a `SKILL.md` file with skill documentation
3. Organize resources into `scripts/`, `references/`, and `assets/` as needed
4. Update `marketplace.json` to include the new skill
5. Test the skill thoroughly before submitting

## Requirements

- Claude AI assistant with plugin support
- For LangChain 1.0 skill:
  - Python 3.8+
  - LangChain 1.0+
  - API keys for model providers (OpenAI, Anthropic, etc.)

## License

MIT License

## Author

**AI Zimo**
- Email: 2876253980@qq.com

## Version History

- **1.0.0** (2025-11-09)
  - Initial release
  - LangChain 1.0 development skill
  - Comprehensive RAG and chain building support
  - Message handling and response analysis capabilities

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: 2876253980@qq.com

---

**Note**: This plugin enhances Claude's capabilities but requires Claude's plugin system to be properly configured and enabled.

