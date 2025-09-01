from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("AUTH_TOKEN")
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token
