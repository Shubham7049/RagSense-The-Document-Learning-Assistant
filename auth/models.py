"""Authentication and user data models for RAG Sense."""

from pydantic import BaseModel, Field, field_validator


class User(BaseModel):
    """Public user model (safe for session state, logging, and UI)."""

    id: int = Field(description="Unique user identifier")
    username: str = Field(description="Unique username or email")
    created_at: str = Field(description="ISO-formatted creation timestamp")


class UserInDB(User):
    """Internal database user model containing the password hash."""

    password_hash: str = Field(description="Hashed password with salt")


class UserRegistration(BaseModel):
    """Input validation model for new user registration."""

    username: str = Field(min_length=3, max_length=50, description="Username or email")
    password: str = Field(min_length=6, max_length=128, description="Plaintext password")
    confirm_password: str = Field(min_length=6, max_length=128, description="Password confirmation")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        return clean

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        password = info.data.get("password")
        if password and v != password:
            raise ValueError("Passwords do not match.")
        return v


class UserLogin(BaseModel):
    """Input validation model for user login."""

    username: str = Field(min_length=1, description="Username or email")
    password: str = Field(min_length=1, description="Plaintext password")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Username cannot be empty.")
        return clean
