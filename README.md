# AI-ZiMo Plugin

## Skills

This plugin includes the following specialized skills:

### Apple Development Skills
- **apple-development-skills:app-store-submission** - Generate App Store submission materials including promotional text, description, and keywords. Supports bilingual submissions (Chinese/English) and ensures compliance with Apple's character limits and guidelines.
- **apple-development-skills:privacy-policy-generator** - Generate comprehensive, legally-compliant privacy policies for applications. Supports automated codebase scanning and interactive questionnaires with GDPR/CCPA compliance addendums.

### Convex Skills
Comprehensive skills for Convex backend development:

#### Core Convex Skills
- **convex-skills:convex-queries** - Building and optimizing Convex queries
- **convex-skills:convex-mutations** - Creating and managing Convex mutations
- **convex-skills:convex-actions-general** - General Convex actions development

#### Convex Agents Skills
- **convex-skills:convex-agents-fundamentals** - Core concepts and fundamentals of Convex agents
- **convex-skills:convex-agents-messages** - Message handling in Convex agents
- **convex-skills:convex-agents-threads** - Thread management for Convex agents
- **convex-skills:convex-agents-tools** - Tool integration and usage in Convex agents
- **convex-skills:convex-agents-context** - Context management for Convex agents
- **convex-skills:convex-agents-rag** - RAG (Retrieval-Augmented Generation) implementation
- **convex-skills:convex-agents-workflows** - Workflow orchestration in Convex agents
- **convex-skills:convex-agents-rate-limiting** - Rate limiting strategies for Convex agents
- **convex-skills:convex-agents-streaming** - Streaming responses in Convex agents
- **convex-skills:convex-agents-files** - File handling in Convex agents
- **convex-skills:convex-agents-human-agents** - Human-in-the-loop agent patterns
- **convex-skills:convex-agents-usage-tracking** - Usage tracking and analytics
- **convex-skills:convex-agents-debugging** - Debugging techniques for Convex agents
- **convex-skills:convex-agents-playground** - Testing and experimentation environment

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