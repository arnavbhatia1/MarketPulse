"""
Streamlit helpers for page-level access control.

Usage at the top of any protected page:

    from app.components.auth_guard import require_auth
    require_auth()            # any logged-in user
    require_auth(premium=True)  # premium users only
"""

import streamlit as st

from src.auth.auth import (
    get_current_user,
    is_guest,
    is_premium as check_premium,
    login_user,
    register_user,
)


def require_auth(premium: bool = False):
    """Gate a page behind authentication.

    Call at the very top of a Streamlit page. If the user is not
    authenticated (and not a guest), shows a warning and halts rendering.
    If *premium* is True and the user lacks a premium account, shows an
    upgrade overlay and halts.
    """
    if is_guest(st.session_state):
        if premium:
            _show_upgrade_overlay()
            st.stop()
        return  # guests pass non-premium gates

    user = get_current_user(st.session_state)
    if user is None:
        st.warning("Please sign in to access this page.")
        show_login_form()
        st.stop()

    if premium and not check_premium(st.session_state):
        _show_upgrade_overlay()
        st.stop()


def show_login_form() -> dict | None:
    """Render a login / register / guest form. Returns user dict on success."""
    tab_signin, tab_register = st.tabs(["Sign In", "Create Account"])

    # ── Sign In ──────────────────────────────────────────────────────
    with tab_signin:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In")

        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
                return None
            user = login_user(email, password)
            if user is None:
                st.error("Invalid email or password.")
                return None
            _set_session(user)
            st.rerun()

    # ── Create Account ───────────────────────────────────────────────
    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            reg_submitted = st.form_submit_button("Create Account")

        if reg_submitted:
            if not reg_email or not reg_password:
                st.error("Email and password are required.")
                return None
            if reg_password != reg_confirm:
                st.error("Passwords do not match.")
                return None
            if len(reg_password) < 8:
                st.error("Password must be at least 8 characters.")
                return None
            try:
                user = register_user(reg_email, reg_password)
            except ValueError as exc:
                st.error(str(exc))
                return None
            _set_session(user)
            st.rerun()

    # ── Guest access ─────────────────────────────────────────────────
    st.divider()
    if st.button("Continue as Guest"):
        st.session_state["guest"] = True
        st.session_state.pop("user_id", None)
        st.rerun()

    return None


# ── Internal helpers ─────────────────────────────────────────────────


def _set_session(user: dict):
    """Persist authentication into Streamlit session state."""
    st.session_state["user_id"] = user["user_id"]
    st.session_state.pop("guest", None)


def _show_upgrade_overlay():
    """Display a premium-required overlay and halt the page."""
    st.warning("This feature requires a MarketPulse Premium account.")
    st.info(
        "Upgrade to Premium to unlock advanced analytics, portfolio "
        "management, and AI-powered insights."
    )
