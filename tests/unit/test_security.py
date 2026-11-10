"""Password hashing, sessions and brute-force limiting."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.security import (
    LoginLimiter,
    SessionStore,
    authenticate,
    hash_password,
    load_user,
    save_user,
    verify_password,
)

PW = "correct-horse-battery"


def test_hash_roundtrip_and_salting() -> None:
    a, b = hash_password(PW), hash_password(PW)
    assert a != b  # random salt
    assert a.startswith("scrypt$") and PW not in a
    assert verify_password(PW, a) and not verify_password(PW + "x", a)


def test_short_password_rejected() -> None:
    with pytest.raises(ValueError):
        hash_password("short")


def test_malformed_hash_is_rejected() -> None:
    assert not verify_password(PW, "garbage")
    assert not verify_password(PW, "md5$1$2$3$4$5")


def test_user_file_stores_only_hash(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    save_user(path, "admin", PW)
    assert PW not in path.read_text(encoding="utf-8")
    user = load_user(path)
    assert user is not None and user.username == "admin"
    assert authenticate(user, "admin", PW)
    assert not authenticate(user, "admin", "wrong-password-123")
    assert not authenticate(user, "someone", PW)
    assert not authenticate(None, "admin", PW)
    assert load_user(tmp_path / "missing.json") is None
    with pytest.raises(ValueError):
        save_user(path, "bad name!", PW)


def test_sessions_expire_and_revoke() -> None:
    store = SessionStore(idle_seconds=10, max_age_seconds=100)
    token = store.create("admin", now=0)
    assert store.get(token, now=5) is not None
    assert store.get(token, now=14) is not None  # idle timer refreshed at t=5
    assert store.get(token, now=30) is None  # idle timeout
    token = store.create("admin", now=0)
    for t in range(5, 100, 5):
        assert store.get(token, now=t) is not None
    assert store.get(token, now=101) is None  # absolute lifetime
    token = store.create("admin", now=0)
    store.revoke(token)
    assert store.get(token, now=1) is None
    assert store.get(None) is None and store.get("forged") is None


def test_session_tokens_not_stored_in_clear() -> None:
    store = SessionStore()
    token = store.create("admin")
    assert token not in store._sessions


def test_login_limiter_locks_and_resets() -> None:
    lim = LoginLimiter(max_failures=3, window=60, lockout=120)
    for t in range(3):
        assert lim.retry_after("ip", now=t) == 0
        lim.failure("ip", now=t)
    assert lim.retry_after("ip", now=3) > 0
    assert lim.retry_after("ip", now=200) == 0
    lim.failure("other", now=0)
    lim.success("other")
    assert lim.retry_after("other", now=0) == 0
