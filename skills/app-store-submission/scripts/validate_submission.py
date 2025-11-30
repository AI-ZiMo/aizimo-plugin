#!/usr/bin/env python3
"""
Validate App Store submission text against character limits.

Usage:
    python validate_submission.py --promotional "Your text" --description "Your desc" --keywords "key1,key2"

Or validate from a file:
    python validate_submission.py --file submission.txt
"""

import argparse
import sys
import re

# App Store character limits
LIMITS = {
    'promotional': 170,
    'description': 4000,
    'keywords': 100,
    'app_name': 30,
    'subtitle': 30
}

def count_characters(text):
    """Count characters in text."""
    return len(text) if text else 0

def check_special_characters(text, field_name):
    """Check for problematic special characters."""
    issues = []

    # Check for emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)

    if emoji_pattern.search(text):
        issues.append(f"Contains emojis (not recommended for {field_name})")

    # Check for special quotation marks
    if '「' in text or '」' in text:
        issues.append(f"Contains special quotation marks 「」 (use standard quotes)")

    # Check for special bullets
    if '•' in text or '✓' in text or '✗' in text:
        issues.append(f"Contains special bullets (•, ✓, ✗) - use hyphens or numbers")

    return issues

def validate_field(text, field_type):
    """Validate a single field against limits."""
    if not text:
        return {
            'valid': False,
            'count': 0,
            'limit': LIMITS.get(field_type, 0),
            'issues': ['Field is empty']
        }

    char_count = count_characters(text)
    limit = LIMITS.get(field_type, 0)
    issues = []

    # Check character limit
    if char_count > limit:
        issues.append(f'Exceeds limit by {char_count - limit} characters')

    # Check for special characters
    special_char_issues = check_special_characters(text, field_type)
    issues.extend(special_char_issues)

    return {
        'valid': len(issues) == 0,
        'count': char_count,
        'limit': limit,
        'issues': issues
    }

def print_validation_result(field_name, result):
    """Print validation result for a field."""
    status = "✅ PASS" if result['valid'] else "❌ FAIL"
    print(f"\n{field_name.upper()}:")
    print(f"  Status: {status}")
    print(f"  Characters: {result['count']}/{result['limit']}")

    if result['count'] <= result['limit']:
        remaining = result['limit'] - result['count']
        print(f"  Remaining: {remaining} characters")

    if result['issues']:
        print(f"  Issues:")
        for issue in result['issues']:
            print(f"    - {issue}")

def main():
    parser = argparse.ArgumentParser(
        description='Validate App Store submission text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_submission.py --promotional "Your promo text"
  python validate_submission.py --keywords "task,todo,habit"
  python validate_submission.py --promotional "Text" --description "Desc" --keywords "keys"
        """
    )

    parser.add_argument('--promotional', help='Promotional text to validate')
    parser.add_argument('--description', help='Description text to validate')
    parser.add_argument('--keywords', help='Keywords to validate')
    parser.add_argument('--app-name', help='App name to validate')
    parser.add_argument('--subtitle', help='Subtitle to validate')

    args = parser.parse_args()

    # Check if at least one field is provided
    if not any([args.promotional, args.description, args.keywords, args.app_name, args.subtitle]):
        parser.print_help()
        sys.exit(1)

    all_valid = True

    # Validate each provided field
    if args.promotional:
        result = validate_field(args.promotional, 'promotional')
        print_validation_result('Promotional Text', result)
        all_valid = all_valid and result['valid']

    if args.description:
        result = validate_field(args.description, 'description')
        print_validation_result('Description', result)
        all_valid = all_valid and result['valid']

    if args.keywords:
        result = validate_field(args.keywords, 'keywords')
        print_validation_result('Keywords', result)
        all_valid = all_valid and result['valid']

    if args.app_name:
        result = validate_field(args.app_name, 'app_name')
        print_validation_result('App Name', result)
        all_valid = all_valid and result['valid']

    if args.subtitle:
        result = validate_field(args.subtitle, 'subtitle')
        print_validation_result('Subtitle', result)
        all_valid = all_valid and result['valid']

    # Exit with appropriate code
    print("\n" + "="*50)
    if all_valid:
        print("✅ All fields passed validation!")
        sys.exit(0)
    else:
        print("❌ Some fields failed validation. Please fix the issues above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
