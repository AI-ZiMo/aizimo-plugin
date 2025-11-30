# AI-ZiMo Plugin

## Skills

This plugin includes the following specialized skills:

### Agent Skills
- **Agent-skills:langchain-1.0** - LangChain 1.0 development support including building chains, implementing RAG systems, integrating models, creating agents, and troubleshooting LangChain applications.

### Apple Development Skills
- **apple-development-skills:app-store-submission** - Generate App Store submission materials including promotional text, description, and keywords. Supports bilingual submissions (Chinese/English) and ensures compliance with Apple's character limits and guidelines.
- **apple-development-skills:privacy-policy-generator** - Generate comprehensive, legally-compliant privacy policies for applications. Supports automated codebase scanning and interactive questionnaires with GDPR/CCPA compliance addendums.

## Installation

Install the plugin in Claude Code:

```bash
claude
/plugin marketplace add AI-ZiMo/aizimo-plugin
```

## Enable Plugin

Remember to enable the plugin after installation.

## Debugging

If you cannot use the skill even after installing the plugin, run `claude --debug` to check whether the skills are loaded from the plugin through looking at the debug logs.