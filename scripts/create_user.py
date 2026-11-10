"""Create or replace the single console account (there is no sign-up).

Only the scrypt hash of the password is stored, in ``secrets/auth.json`` (git-ignored).

Usage:
    python scripts/create_user.py --username admin            # prompts for the password
    python scripts/create_user.py --username admin --generate # prints a random password once
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.security import save_user  # noqa: E402


def generate_password(length: int = 20) -> str:
    """Random password with letters, digits and symbols."""
    alphabet = string.ascii_letters + string.digits + "-_.!@#%"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.isdigit() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.islower() for c in pw)
        ):
            return pw


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--generate", action="store_true", help="generate a random password")
    args = parser.parse_args()
    path = Path(os.getenv("FAULTAWARE_AUTH_FILE", ROOT / "secrets" / "auth.json"))
    if args.generate:
        password = generate_password()
    else:
        password = getpass.getpass("New password (min 12 chars): ")
        if password != getpass.getpass("Repeat password: "):
            sys.exit("Passwords do not match.")
    save_user(path, args.username, password)
    print(f"Account '{args.username}' saved to {path} (scrypt hash only).")
    if args.generate:
        print(f"Generated password (shown once, store it safely): {password}")
    print("Restart the API for the change to take effect.")


if __name__ == "__main__":
    main()
