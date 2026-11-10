"""Authentication and hardening for the demo API (standard library only).

* Passwords: scrypt (memory-hard KDF) with a per-user random salt; verified in constant time.
  A dummy hash is checked for unknown usernames so timing does not reveal valid names.
* Sessions: 256-bit random tokens in an HttpOnly, SameSite=Strict cookie. Only the SHA-256 of a
  token is kept server-side; sessions expire after an idle timeout and an absolute lifetime and
  are revoked on logout.
* Brute force: after ``max_failures`` failed logins from one client within ``window`` seconds,
  further attempts are refused for ``lockout`` seconds (HTTP 429).
* Single user, no sign-up: the account lives in a JSON file created by
  ``scripts/create_user.py`` (git-ignored ``secrets/auth.json``).

Transport encryption is TLS: run behind HTTPS in any shared deployment (see README).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRYPT_N, SCRYPT_R, SCRYPT_P, KEY_LEN = 2**15, 8, 1, 32
_MAXMEM = 128 * SCRYPT_N * SCRYPT_R * 2


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str, *, n: int = SCRYPT_N) -> str:
    """Hash a password as ``scrypt$n$r$p$salt$key`` (salt and key base64)."""
    if len(password) < 12:
        raise ValueError("password must be at least 12 characters")
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=n,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_LEN,
        maxmem=_MAXMEM * max(1, n // SCRYPT_N),
    )
    return f"scrypt${n}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against an encoded scrypt hash."""
    try:
        scheme, n, r, p, salt, key = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(key)
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
            maxmem=128 * int(n) * int(r) * 2,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True)
class User:
    """The single provisioned account."""

    username: str
    password_hash: str


def load_user(path: Path) -> User | None:
    """Read the account file; ``None`` if it does not exist."""
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return User(raw["username"], raw["password_hash"])


def save_user(path: Path, username: str, password: str) -> User:
    """Create/replace the account file (only the scrypt hash is stored)."""
    if not username or len(username) > 64 or not username.replace("_", "").isalnum():
        raise ValueError("username must be 1-64 letters, digits or underscores")
    user = User(username, hash_password(password))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"username": user.username, "password_hash": user.password_hash}),
        encoding="utf-8",
    )
    with contextlib.suppress(OSError):  # Windows ignores POSIX modes
        path.chmod(0o600)
    return user


_dummy_hash: str | None = None


def _dummy() -> str:
    """A real-cost hash used when the username is wrong (equalises response time)."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_urlsafe(16))
    return _dummy_hash


def authenticate(user: User | None, username: str, password: str) -> bool:
    """True iff the credentials match; always does one full-cost scrypt evaluation."""
    if user is None or not hmac.compare_digest(username.encode(), user.username.encode()):
        verify_password(password, _dummy())
        return False
    return verify_password(password, user.password_hash)


@dataclass
class Session:
    """Server-side session record (keyed by the token's SHA-256)."""

    username: str
    created: float
    last_seen: float


@dataclass
class SessionStore:
    """In-memory session store with idle and absolute expiry."""

    idle_seconds: float = 3600.0
    max_age_seconds: float = 8 * 3600.0
    _sessions: dict[str, Session] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, username: str, now: float | None = None) -> str:
        """Start a session and return its secret token."""
        now = time.time() if now is None else now
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[self._key(token)] = Session(username, now, now)
        return token

    def get(self, token: str | None, now: float | None = None) -> Session | None:
        """Return the live session for ``token`` (refreshing idle time) or ``None``."""
        if not token:
            return None
        now = time.time() if now is None else now
        key = self._key(token)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                return None
            if (
                now - session.last_seen > self.idle_seconds
                or now - session.created > self.max_age_seconds
            ):
                del self._sessions[key]
                return None
            session.last_seen = now
            return session

    def revoke(self, token: str | None) -> None:
        """End a session (no-op if unknown)."""
        if token:
            with self._lock:
                self._sessions.pop(self._key(token), None)


@dataclass
class LoginLimiter:
    """Fixed-window brute-force protection per client key."""

    max_failures: int = 5
    window: float = 900.0
    lockout: float = 900.0
    _failures: dict[str, list[float]] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def retry_after(self, key: str, now: float | None = None) -> int:
        """Seconds until ``key`` may try again (0 = allowed)."""
        now = time.time() if now is None else now
        with self._lock:
            return max(0, int(self._locked_until.get(key, 0) - now))

    def failure(self, key: str, now: float | None = None) -> None:
        """Record a failed attempt; lock the key when the threshold is reached."""
        now = time.time() if now is None else now
        with self._lock:
            recent = [t for t in self._failures.get(key, []) if now - t < self.window]
            recent.append(now)
            self._failures[key] = recent
            if len(recent) >= self.max_failures:
                self._locked_until[key] = now + self.lockout
                self._failures[key] = []

    def success(self, key: str) -> None:
        """Clear the failure history after a successful login."""
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store",
}
