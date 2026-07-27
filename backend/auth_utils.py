import os
from datetime import datetime, timedelta, timezone
import bcrypt
from jose import jwt, JWTError
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-for-dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ──────────────────────────────────────────────────────────
# RBAC – Role Hierarchy
# ──────────────────────────────────────────────────────────

ROLE_HIERARCHY: dict[str, int] = {
    "Student":      1,
    "Other":        1,
    "Professional": 2,
    "Researcher":   2,
    "Professor":    3,
}

def get_role_level(role: str) -> int:
    """Return the numeric permission level for a role (higher = more access)."""
    return ROLE_HIERARCHY.get(role, 1)


# ──────────────────────────────────────────────────────────
# Auth Dependencies
# ──────────────────────────────────────────────────────────

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Require a valid JWT. Returns username string."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid auth credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid auth credentials")

def get_current_user_with_role(token: str = Depends(oauth2_scheme)) -> dict:
    """Require a valid JWT. Returns {'username': str, 'role': str}."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "Student")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid auth credentials")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid auth credentials")

def require_min_role(min_role: str):
    """
    Dependency factory – guards a route to users whose role level >= min_role level.

    Usage:
        @app.post("/api/compare")
        async def compare(user = Depends(require_min_role("Researcher"))):
            ...
    """
    min_level = get_role_level(min_role)

    def checker(user: dict = Depends(get_current_user_with_role)) -> dict:
        user_level = get_role_level(user.get("role", "Student"))
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Access denied. This feature requires '{min_role}' role or higher. "
                    f"Your current role: '{user.get('role')}'. "
                    f"Please contact support to upgrade your account."
                ),
            )
        return user

    return checker

# Optional auth - if token provided, decode it, else return None
def get_current_user_optional(token: str = Security(oauth2_scheme)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
