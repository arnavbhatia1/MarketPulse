"""
Authentication module for MarketPulse.

Bcrypt password hashing + Streamlit session state management.
No OAuth, no email verification, no password reset -- just the basics.
"""

import re

import bcrypt

from src.storage.db import create_user, get_user_by_email, get_user_by_id

# Minimal email regex -- catches obvious typos, not RFC 5322 compliant.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    """Return a bcrypt hash of the given password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def register_user(email: str, password: str) -> dict:
    """Validate, hash, persist, and return the new user dict.

    Raises:
        ValueError: If the email format is invalid or already registered.
    """
    email = email.strip().lower()

    if not _EMAIL_RE.match(email):
        raise ValueError("Invalid email format.")

    if get_user_by_email(email) is not None:
        raise ValueError("An account with this email already exists.")

    pw_hash = hash_password(password)
    user_id = create_user(email, pw_hash)

    # create_user returns a user_id string; fetch the full row so callers
    # get a consistent dict with all columns.
    return get_user_by_id(user_id)


def login_user(email: str, password: str) -> dict | None:
    """Authenticate by email + password. Returns user dict or None."""
    email = email.strip().lower()
    user = get_user_by_email(email)
    if user is None:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def get_current_user(session_state) -> dict | None:
    """Look up the logged-in user from Streamlit session state."""
    user_id = session_state.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def is_premium(session_state) -> bool:
    """Return True if the current user has a premium account."""
    user = get_current_user(session_state)
    if user is None:
        return False
    return user.get("is_premium", 0) == 1


def is_guest(session_state) -> bool:
    """Return True if the session is in guest mode."""
    return session_state.get("guest", False)
