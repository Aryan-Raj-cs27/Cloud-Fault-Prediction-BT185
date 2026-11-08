# Security policy

## Reporting a vulnerability
Please **do not open a public issue** for security problems. Report them privately through GitHub: on this repository open **Security → Report a vulnerability** (private vulnerability reporting). Include the steps to reproduce and the affected version or commit.

The team aims to acknowledge reports within 7 days. This is a student research project, so there is no bug bounty.

## Supported versions
Only the latest commit on `main` receives fixes.

## Deployment checklist
- Create the console account on the server with `python scripts/create_user.py --username admin`. Never commit `secrets/`.
- Serve over HTTPS (the provided `docker-compose.yml` uses Caddy) and set `FAULTAWARE_COOKIE_SECURE=1`.
- Keep `FAULTAWARE_DOCS` unset in production so the API docs stay disabled.
- Install model artefacts only from a release bundle whose SHA-256 you have verified (`scripts/install_artifacts.py` checks it).

The security design (password hashing, sessions, rate limiting, origin checks, headers) is described in the README's *Security model* section.
