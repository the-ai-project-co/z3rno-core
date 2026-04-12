# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Z3rno, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email **security@z3rno.dev** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge your report within 48 hours and aim to release a fix within 7 days for critical vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x.x   | Yes (current development) |

## Security Best Practices

- Never commit API keys, secrets, or credentials
- Use environment variables for all sensitive configuration
- Enable Row-Level Security (RLS) in production
- Rotate API keys regularly
- Use HTTPS in production
