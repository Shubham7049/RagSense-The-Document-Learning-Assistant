"""Authentication interface for RAG Sense (Login & Registration)."""

import streamlit as st

from auth.service import AuthService
from auth.session import login_user
from config.settings import settings
from core.exceptions import AuthenticationError


def render_auth_view() -> None:
    """Render the clean authentication screen with Login and Register tabs."""
    auth_service = AuthService()

    # Center container styling
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;">
                <h1 style="margin-bottom: 0.2rem; font-weight: 800; font-size: 2.6rem;">{settings.app_name}</h1>
                <p style="color: #64748b; font-size: 1.15rem;">Welcome back.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            st.write("")
            with st.form(key="login_form", clear_on_submit=False):
                username = st.text_input("Username or Email", key="login_username", placeholder="Enter your username")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                    placeholder="Enter your password",
                )
                submit_login = st.form_submit_button("Log In", use_container_width=True)

                if submit_login:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        try:
                            user = auth_service.authenticate(username=username, password=password)
                            login_user(user)
                            st.success(f"Welcome back, {user.username}!")
                            st.rerun()
                        except AuthenticationError as auth_err:
                            st.error(auth_err.message)
                        except Exception as e:
                            st.error(f"An unexpected error occurred during login: {e}")

        with tab_register:
            st.write("")
            with st.form(key="register_form", clear_on_submit=False):
                new_username = st.text_input(
                    "Username or Email",
                    key="reg_username",
                    placeholder="Choose a unique username",
                )
                new_password = st.text_input(
                    "Password",
                    type="password",
                    key="reg_password",
                    placeholder="Minimum 6 characters",
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="reg_confirm_password",
                    placeholder="Re-enter your password",
                )
                submit_register = st.form_submit_button("Create Account", use_container_width=True)

                if submit_register:
                    try:
                        user = auth_service.register(
                            username=new_username,
                            password=new_password,
                            confirm_password=confirm_password,
                        )
                        # Auto-login newly registered user
                        login_user(user)
                        st.success(f"Account created successfully! Welcome, {user.username}.")
                        st.rerun()
                    except AuthenticationError as auth_err:
                        st.error(auth_err.message)
                    except Exception as e:
                        st.error(f"Registration failed: {e}")
