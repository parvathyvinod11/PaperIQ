from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime, timezone
import sys
import os

# Add parent dir to path so we can import from database
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import users_collection
from auth_utils import get_password_hash, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/signup")
async def signup(request: SignupRequest):
    if not request.username or not request.password or not request.email:
        raise HTTPException(status_code=400, detail="Username, email, and password required")
        
    existing_user = await users_collection.find_one({"$or": [{"username": request.username}, {"email": request.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    hashed_password = get_password_hash(request.password)
    user_doc = {
        "username": request.username,
        "email": request.email,
        "password_hash": hashed_password,
        "role": request.role,
        "created_at": datetime.now(timezone.utc)
    }
    
    await users_collection.insert_one(user_doc)
    return {"message": "User created successfully"}

@router.post("/login")
async def login(request: LoginRequest):
    # Lookup by email now instead of username
    user = await users_collection.find_one({"email": request.email})
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(data={"sub": user["username"], "role": user.get("role")})
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "username": user["username"], 
        "email": user["email"],
        "role": user.get("role")
    }
