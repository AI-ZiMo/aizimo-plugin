# California Privacy Rights (CCPA/CPRA)

For California residents, the following additional terms apply under the California Consumer Privacy Act (CCPA) and California Privacy Rights Act (CPRA):

## Categories of Personal Information We Collect

{{#PI_CATEGORIES}}
- **{{CATEGORY_NAME}}:** {{CATEGORY_EXAMPLES}}
{{/PI_CATEGORIES}}

## Your California Privacy Rights

California residents have the right to:

1. **Right to Know:** Request disclosure of:
   - Categories of personal information collected
   - Categories of sources from which information is collected
   - Business or commercial purpose for collection
   - Categories of third parties with whom we share information
   - Specific pieces of personal information collected

2. **Right to Delete:** Request deletion of personal information we have collected, subject to certain exceptions

3. **Right to Correct:** Request correction of inaccurate personal information

4. **Right to Opt-Out:** Opt-out of the "sale" or "sharing" of personal information

5. **Right to Limit:** Limit the use and disclosure of sensitive personal information

6. **Right to Non-Discrimination:** Not receive discriminatory treatment for exercising your privacy rights

## How to Exercise Your Rights

To exercise your CCPA rights:
- Email: {{CONTACT_EMAIL}}
{{#CCPA_WEBFORM}}- Web Form: {{CCPA_WEBFORM_URL}}{{/CCPA_WEBFORM}}
{{#CCPA_PHONE}}- Phone: {{CCPA_PHONE_NUMBER}}{{/CCPA_PHONE}}

We will respond to verifiable requests within 45 days.

## Verification Process

To verify your identity, we may request:
- {{VERIFICATION_METHOD_1}}
- {{VERIFICATION_METHOD_2}}

## Authorized Agent

You may designate an authorized agent to make requests on your behalf. The agent must provide:
- Written permission from you
- Proof of their authority

## Do Not Sell or Share My Personal Information

{{#SELLS_OR_SHARES_DATA}}
We may "sell" or "share" the following categories of personal information for targeted advertising purposes:
{{#SOLD_CATEGORIES}}
- {{CATEGORY}}
{{/SOLD_CATEGORIES}}

To opt-out, click here: [Do Not Sell or Share My Personal Information]({{OPT_OUT_LINK}})
{{/SELLS_OR_SHARES_DATA}}
{{^SELLS_OR_SHARES_DATA}}
We do not "sell" or "share" personal information as defined by CCPA.
{{/SELLS_OR_SHARES_DATA}}

## Sensitive Personal Information

{{#COLLECTS_SENSITIVE}}
We collect the following categories of sensitive personal information:
{{#SENSITIVE_CATEGORIES}}
- {{CATEGORY}}
{{/SENSITIVE_CATEGORIES}}

We use and disclose sensitive personal information only for purposes permitted by CCPA.
{{/COLLECTS_SENSITIVE}}

## Data Retention

We retain personal information for the following periods:
{{#RETENTION_TABLE}}
- {{DATA_TYPE}}: {{PERIOD}}
{{/RETENTION_TABLE}}

## Metrics (Annual Disclosure)

In the preceding 12 months:
- Requests to Know: {{REQUESTS_TO_KNOW}} ({{REQUESTS_TO_KNOW_GRANTED}} granted)
- Requests to Delete: {{REQUESTS_TO_DELETE}} ({{REQUESTS_TO_DELETE_GRANTED}} granted)
- Requests to Opt-Out: {{REQUESTS_TO_OPT_OUT}}
- Average response time: {{AVERAGE_RESPONSE_TIME}} days
