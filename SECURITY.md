# Security Policy

## Credentials

Never store API tokens, passwords, private keys, cookies, or credential files in this repository. Authenticate interactively with standard input or an approved secret manager. Treat any token pasted into chat, a notebook, a terminal transcript, or an issue as compromised and rotate it immediately.

## Data handling

Challenge controls, support matrices, generated single-cell predictions, and submission containers stay on approved storage. Before sharing any derived artifact, verify the source license and remove personal or operational metadata.

## Reporting

Report a suspected credential or data exposure privately to the repository owner. Do not open a public issue. Rotate affected credentials first, then remove the material from working trees and Git history before any further push.
