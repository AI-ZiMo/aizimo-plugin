---
name: app-store-submission
description: Generate App Store submission materials including promotional text, description, and keywords. Use when users need to create or update App Store Connect content for iOS/macOS apps, require text that meets Apple's character limits (promotional text 170 chars, description 4000 chars, keywords 100 chars), need bilingual submissions (Chinese/English), or want to ensure compliance with App Store guidelines. Also use for Google Play Store submissions with similar requirements.
---

# App Store Submission

Create compelling, compliant App Store submission materials that meet Apple's requirements and attract users.

## Workflow

Follow this process to create complete App Store submission materials:

### Step 1: Understand the App

Before writing any content, gather essential information:

- What does the app do? (core functionality)
- What problem does it solve?
- Who is the target audience?
- What are the key features? (3-6 main features)
- What makes it unique from competitors?
- What privacy/security features does it have?
- Does it support multiple languages?
- Any special technologies or integrations?

Ask clarifying questions if the user hasn't provided this information.

### Step 2: Create Content in This Order

Create content in this sequence to maintain consistency:

1. **Keywords** - Define core terms first to guide other content
2. **Description** - Write full description using keywords naturally
3. **Promotional Text** - Distill key message from description

### Step 3: Write Keywords

Create a keyword list that maximizes discoverability:

**Character limit: 100 characters maximum**

Guidelines:
- Separate keywords with commas (English or Chinese commas both work)
- No spaces needed between keywords
- Prioritize most important keywords first
- Include category-relevant terms
- Use both broad and specific terms
- For bilingual apps, include keywords in both languages
- Avoid keyword stuffing or repetition

Example approach:
```
app category,main feature,secondary feature,use case,target audience,unique aspect
```

Chinese example:
```
任务管理,待办事项,习惯养成,效率工具,游戏化,特工,目标追踪,自我提升
```

English example:
```
task,todo,habit,tracker,productivity,gamification,agent,mission,goals
```

### Step 4: Write Description

Create a compelling description that converts readers to downloaders:

**Character limit: 4,000 characters maximum**

Structure:
1. **Hook** (2-3 sentences): Capture attention, state the main benefit
2. **Core Features** (structured list): 3-6 key features with brief explanations
3. **Why Choose This App** (bullet points): Unique value propositions
4. **Target Audience** (optional): Who benefits most
5. **Call to Action** (1 sentence): Encourage download

Guidelines:
- First 3 lines are most important (visible without tapping "more")
- Use clear section headers for scannability
- Focus on benefits, not just features
- Avoid emojis and special characters (「」✓•)
- Emphasize privacy/security if relevant
- Use active voice
- Be specific, not generic
- Include localized content for each market

### Step 5: Write Promotional Text

Create a concise, impactful message:

**Character limit: 170 characters maximum**

Guidelines:
- Lead with the strongest current feature or update
- Can be updated anytime without new submission
- Should complement, not duplicate, the description
- Create urgency or highlight exclusivity
- Avoid emojis and special characters
- Test readability in both languages if bilingual

Example structure:
```
[Action verb] + [key feature]! [Supporting benefit]. [Privacy/unique value]. [Call to action].
```

### Step 6: Validate Content

Use the validation script to check all content:

```bash
python scripts/validate_submission.py \
  --promotional "Your promotional text" \
  --description "Your description" \
  --keywords "your,keywords,here"
```

The script checks:
- Character count vs limits
- Presence of emojis
- Special characters (「」, •, ✓)
- Provides character count and remaining characters

Fix any issues flagged by the validator.

### Step 7: Format for Different Markets

If creating bilingual submissions:

1. Create complete content in both languages (not direct translations)
2. Adapt messaging for cultural context:
   - **Chinese**: Emphasize privacy, gamification, local storage, no data collection
   - **English**: Focus on convenience, time-saving, productivity, security
3. Ensure both versions meet character limits independently
4. Validate each language version separately

## Content Guidelines

### Always Avoid

- Emojis in promotional text and description
- Special quotation marks: 「」
- Special bullets: •, ✓, ✗
- Markdown formatting symbols: **, ##, -
- Pricing information (use in-app purchase details instead)
- Competitor app references
- External links (except privacy policy/terms)

### Always Include

- Clear value proposition in first paragraph
- Privacy/security information if relevant
- Key differentiators from competitors
- Target audience context
- Benefit-focused language

## Resources

### Reference Documentation

**references/app-store-guidelines.md** - Comprehensive guidelines including:
- Detailed character limits for all fields
- Content guidelines and best practices
- Localization tips for Chinese and English markets
- Common mistakes to avoid

Read this file when:
- Unsure about specific requirements
- Need localization guidance
- Want to understand best practices in depth
- Troubleshooting validation issues

### Validation Script

**scripts/validate_submission.py** - Automated validation tool that checks:
- Character counts against limits
- Problematic special characters
- Emojis in content
- Provides actionable feedback

Run this script after creating content to ensure compliance.

## Examples

### Example 1: Task Management App

**Keywords** (45 chars):
```
任务管理,待办事项,习惯养成,效率工具,游戏化,特工,目标追踪,自我提升,个人成长,小胜利
```

**Promotional Text** (79 chars):
```
化身特工，征服每日任务！全新语音播报功能，让总部为你解读任务详情。记录每个小胜利，见证自己的成长轨迹。无需注册，数据本地存储，你的隐私由你守护。
```

**Description** (excerpt):
```
把枯燥的待办清单变成特工的秘密任务

伟大的我是一款独特的任务管理应用，将日常琐事转化为激动人心的特工任务。在这里，你不再是拖延者，而是接受总部指令、完成重要任务的精英特工。

核心功能

特工风格任务管理
以特工任务的形式管理你的日常事务，每项任务都是总部下达的重要指令，让平凡的工作变得充满使命感。

每日小胜利记录
专属的小胜利任务类型，帮助你记录每天的进步和成就。无论大小，每个胜利都值得被铭记。
...
```

### Example 2: Productivity App (English)

**Keywords** (85 chars):
```
task,todo,habit,tracker,productivity,gamification,agent,mission,goals,self-improvement
```

**Promotional Text** (145 chars):
```
Transform tasks into agent missions! Voice briefings, victory tracking, and complete privacy. No registration required, all data stored locally.
```

**Description** (excerpt):
```
Transform boring to-do lists into secret agent missions

The Great Me is a unique task management app that transforms daily chores into exciting agent missions. Here, you're no longer a procrastinator, but an elite agent receiving headquarters directives and completing important missions.

Core Features

Agent-Style Task Management
Manage your daily tasks as agent missions. Each task is an important directive from headquarters, bringing a sense of purpose to ordinary work.
...
```

## Common Patterns

### Pattern 1: Feature-Heavy Apps

For apps with many features:
- List 3-6 most important features only in description
- Use clear feature headers
- Group related features together
- Mention "and more" to imply additional features

### Pattern 2: Privacy-Focused Apps

For apps emphasizing privacy:
- Mention "no registration," "local storage," "no data collection" prominently
- Include in both promotional text and description
- Add privacy keywords
- Highlight in "Why Choose" section

### Pattern 3: Gamification Apps

For apps using gamification:
- Use engaging, action-oriented language
- Describe the "transformation" or "journey"
- Include gamification keywords
- Appeal to achievement motivation

## Tips for Success

1. **Test readability**: Read content aloud to check flow
2. **Check competitors**: Research similar apps for differentiation
3. **Prioritize mobile**: Most users read on small screens
4. **Update promotional text**: Refresh every major release
5. **A/B test**: Try different descriptions to see what converts
6. **Localize properly**: Avoid direct translations, adapt culturally
7. **Validate early**: Check limits before writing full content
