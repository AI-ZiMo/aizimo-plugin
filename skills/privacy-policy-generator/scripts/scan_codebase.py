#!/usr/bin/env python3
"""
Scans codebase to detect data collection patterns and third-party services.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set

class CodebaseScanner:
    """Scans code for privacy-relevant patterns."""

    # Third-party services patterns
    SERVICES = {
        'analytics': [
            r'google-analytics|gtag|GoogleAnalytics|ga\(',
            r'mixpanel|Mixpanel',
            r'segment|analytics\.track',
            r'amplitude|Amplitude',
            r'firebase/analytics|FirebaseAnalytics',
            r'plausible|Plausible',
        ],
        'ads': [
            r'google-ads|googleads|AdMob',
            r'facebook-ads|FacebookAds',
            r'adservice|AdService',
        ],
        'authentication': [
            r'firebase/auth|FirebaseAuth',
            r'auth0|Auth0',
            r'supabase/auth',
            r'@clerk/clerk',
            r'next-auth|NextAuth',
        ],
        'payment': [
            r'stripe|Stripe',
            r'paypal|PayPal',
            r'square|Square',
            r'braintree|Braintree',
        ],
        'database': [
            r'firebase/firestore|Firestore',
            r'supabase',
            r'mongodb|MongoDB',
            r'postgresql|postgres',
            r'mysql|MySQL',
        ],
        'cloud_storage': [
            r'aws-sdk|AWS\.S3',
            r'@google-cloud/storage',
            r'firebase/storage',
            r'cloudinary|Cloudinary',
        ],
        'crash_reporting': [
            r'sentry|Sentry',
            r'bugsnag|Bugsnag',
            r'crashlytics|Crashlytics',
        ],
        'email': [
            r'sendgrid|SendGrid',
            r'mailgun|Mailgun',
            r'resend|Resend',
            r'nodemailer',
        ],
    }

    # Data collection patterns
    DATA_PATTERNS = {
        'location': r'(geolocation|navigator\.geolocation|location\.coords|getCurrentPosition)',
        'camera': r'(navigator\.camera|getUserMedia.*video|CameraManager)',
        'microphone': r'(navigator\.mediaDevices|getUserMedia.*audio|AudioRecord)',
        'contacts': r'(ContactsContract|CNContact|navigator\.contacts)',
        'storage': r'(localStorage|sessionStorage|AsyncStorage|SharedPreferences)',
        'cookies': r'(document\.cookie|Cookies\.set|setCookie|cookie-parser)',
        'device_id': r'(device\.uuid|deviceId|getDeviceId|advertisingId)',
        'ip_address': r'(request\.ip|req\.socket\.remoteAddress|X-Forwarded-For)',
    }

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.findings: Dict[str, Set[str]] = {
            'services': set(),
            'data_types': set(),
            'files_scanned': set(),
        }

    def scan(self) -> Dict:
        """Scan the codebase and return findings."""
        # Scan package files first
        self._scan_dependencies()

        # Scan source code
        self._scan_source_files()

        return {
            'services': list(self.findings['services']),
            'data_types': list(self.findings['data_types']),
            'total_files_scanned': len(self.findings['files_scanned']),
        }

    def _scan_dependencies(self):
        """Scan package.json, requirements.txt, etc. for dependencies."""
        # Check package.json
        package_json = self.project_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    all_deps = {
                        **data.get('dependencies', {}),
                        **data.get('devDependencies', {})
                    }
                    self._check_dependencies(all_deps)
            except Exception:
                pass

        # Check requirements.txt
        requirements = self.project_path / 'requirements.txt'
        if requirements.exists():
            try:
                with open(requirements) as f:
                    deps = {line.split('==')[0].split('>=')[0].strip(): '1.0'
                           for line in f if line.strip() and not line.startswith('#')}
                    self._check_dependencies(deps)
            except Exception:
                pass

    def _check_dependencies(self, dependencies: Dict[str, str]):
        """Check dependencies against known services."""
        dep_names = ' '.join(dependencies.keys()).lower()

        for category, patterns in self.SERVICES.items():
            for pattern in patterns:
                if re.search(pattern, dep_names, re.IGNORECASE):
                    self.findings['services'].add(category)

    def _scan_source_files(self):
        """Scan source code files for patterns."""
        extensions = {'.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.kt', '.swift', '.m'}

        for ext in extensions:
            for file_path in self.project_path.rglob(f'*{ext}'):
                # Skip node_modules, venv, etc.
                if any(part.startswith('.') or part in {'node_modules', 'venv', 'build', 'dist'}
                       for part in file_path.parts):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self._scan_content(content)
                        self.findings['files_scanned'].add(str(file_path.relative_to(self.project_path)))
                except Exception:
                    pass

    def _scan_content(self, content: str):
        """Scan file content for data collection patterns."""
        # Check for services
        for category, patterns in self.SERVICES.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self.findings['services'].add(category)

        # Check for data patterns
        for data_type, pattern in self.DATA_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                self.findings['data_types'].add(data_type)


def main():
    """Main entry point for CLI usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: scan_codebase.py <project_path>")
        sys.exit(1)

    project_path = sys.argv[1]
    scanner = CodebaseScanner(project_path)
    results = scanner.scan()

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
