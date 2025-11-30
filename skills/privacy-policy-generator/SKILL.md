---
name: privacy-policy-generator
description: Generate comprehensive, legally-compliant privacy policies for applications. Use when the user requests creation of a privacy policy, needs to understand data collection practices, wants GDPR/CCPA compliance, or asks to document privacy practices for mobile apps, web apps, or desktop applications. Supports automated codebase scanning combined with interactive questionnaires.
---

# Privacy Policy Generator

Generate privacy policies for applications by analyzing codebases and gathering information through structured questionnaires. Produces compliant policies with optional GDPR, CCPA, and COPPA addendums.

## Workflow

Follow this three-phase process:

### Phase 1: Discovery

Understand what data the application collects and how it's used.

**1. Scan the codebase:**

```bash
python scripts/scan_codebase.py <project_path>
```

This detects:
- Third-party services (analytics, authentication, payments, etc.)
- Data collection patterns (location, cookies, storage, device info)
- Dependencies that indicate privacy-relevant features

Review the scan output to understand automated findings.

**2. Conduct the questionnaire:**

Use `references/questionnaire.md` to gather additional information. The questionnaire is organized into sections:
- Basic Information (app name, developer, contact)
- Data Collection (what data is collected)
- Data Usage (why data is collected)
- Data Sharing (with whom data is shared)
- User Rights (access, deletion, export)
- Security measures
- Compliance requirements

Ask questions selectively based on:
- What the codebase scan already revealed
- What remains unclear from automated detection
- Which compliance frameworks apply (GDPR, CCPA, COPPA)

**Example interaction:**
```
Claude: "I scanned your codebase and found:
- Google Analytics integration
- Firebase Authentication
- Local storage usage

Let me ask a few clarifying questions:
1. Do you collect users' email addresses through Firebase Auth?
2. What do you use Google Analytics data for?
3. How long do you retain user data?"
```

### Phase 2: Policy Generation

Build the privacy policy using the template system.

**1. Select the base template:**

Start with `assets/templates/basic_policy.md` which uses mustache-style variables:

```
{{APP_NAME}} - Required
{{EFFECTIVE_DATE}} - Required
{{CONTACT_EMAIL}} - Required
{{APP_TYPE}} - Required (e.g., "mobile app", "web application")
```

**2. Populate conditional sections:**

The template includes conditional blocks ({{#CONDITION}}...{{/CONDITION}}). Include sections only when applicable:

```markdown
{{#COLLECTS_ACCOUNT_INFO}}
**Account Information:** We collect: {{ACCOUNT_INFO_LIST}}
{{/COLLECTS_ACCOUNT_INFO}}
```

Set variables based on questionnaire responses:
- Set condition variables (e.g., `COLLECTS_ACCOUNT_INFO`) to true/false
- Populate list variables (e.g., `ACCOUNT_INFO_LIST`) with comma-separated items
- Fill descriptive variables with complete sentences

**3. Add compliance addendums:**

If the app has EU users, append `assets/templates/gdpr_addendum.md`
If the app has California users, append `assets/templates/ccpa_addendum.md`

### Phase 3: Review and Output

**1. Review for completeness:**

Verify the policy covers:
- All data types identified in the scan
- All services found in dependencies
- All user rights applicable to the jurisdiction
- Clear contact information
- Effective date

**2. Generate output formats:**

Default: Markdown format suitable for documentation

Optional formats based on user request:
- **HTML:** Wrap in basic HTML structure with semantic tags
- **Plain text:** Strip markdown formatting for app store submissions
- **PDF:** Use a document generation skill if available

**3. Provide guidance:**

After generating the policy, remind the user to:
- Have legal counsel review before publishing
- Update the effective date when publishing
- Notify users of material changes
- Review annually or when data practices change

## Examples

**Example 1: Simple web app**
```
User: "Create a privacy policy for my todo list web app"

Claude:
1. Scans codebase → finds localStorage, no third-party services
2. Asks: "Do users create accounts?" → No
3. Asks: "Where are users located?" → Global
4. Generates basic policy covering local storage only
```

**Example 2: Mobile app with analytics**
```
User: "I need a privacy policy for my fitness tracking app"

Claude:
1. Scans codebase → finds Firebase Analytics, Google Ads, location API
2. Asks targeted questions about:
   - Health data handling
   - Ad personalization
   - Location precision
   - Age restrictions
3. User indicates: EU and California users, 13+ age requirement
4. Generates policy with GDPR and CCPA addendums
```

## Variable Reference

Core variables for `basic_policy.md`:

**Required:**
- `APP_NAME`: Application name
- `APP_TYPE`: Type (mobile app, web application, etc.)
- `EFFECTIVE_DATE`: When policy takes effect
- `CONTACT_EMAIL`: Privacy contact email

**Conditional sections:**
- `COLLECTS_ACCOUNT_INFO`: Set to true if collecting account data
- `COLLECTS_PAYMENT_INFO`: Set to true if processing payments
- `USES_COOKIES`: Set to true if using cookies
- `SHARES_WITH_SERVICE_PROVIDERS`: Set to true if sharing with vendors
- `RIGHT_ACCESS`: Set to true if users can request data
- `RIGHT_DELETION`: Set to true if users can request deletion

**Lists (comma-separated):**
- `ACCOUNT_INFO_LIST`: "Email addresses, usernames, passwords"
- `USAGE_DATA_LIST`: "Pages visited, features used, time spent"
- `SERVICE_PROVIDERS`: Provider details with names and purposes

**Descriptions (full sentences):**
- `RETENTION_DESCRIPTION`: "We retain data until you delete your account, or for 90 days after your last login."
- `SECURITY_MEASURES`: List of security practices

For complete variable listings, review the template files directly.

## Best Practices

**Be specific:** "We collect your email address and username" beats "We collect personal information"

**Be transparent:** If you use data for advertising, state it clearly

**Match actual practice:** The policy must reflect what the code actually does

**Regional compliance:**
- GDPR requires legal basis for processing (consent, contract, legitimate interest)
- CCPA requires disclosure of "sale" even if no money exchanged
- COPPA requires parental consent for users under 13

**Update triggers:**
- Adding new third-party services
- Collecting new types of data
- Changing data retention periods
- Expanding to new jurisdictions

## Resources

- **scripts/scan_codebase.py**: Automated codebase scanner for data collection detection
- **references/questionnaire.md**: Comprehensive questionnaire (50 questions across 10 categories)
- **assets/templates/**: Privacy policy templates
  - `basic_policy.md`: Base template with mustache variables
  - `gdpr_addendum.md`: GDPR compliance addendum
  - `ccpa_addendum.md`: CCPA/CPRA compliance addendum
