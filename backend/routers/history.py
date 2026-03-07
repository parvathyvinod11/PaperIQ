from fastapi import APIRouter, Depends
from typing import List
from database import history_collection
from auth_utils import get_current_user
from bson import json_util
import json

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("/")
async def get_history(username: str = Depends(get_current_user)):
    """Fetch all past analysis for the current user."""
    cursor = history_collection.find({"username": username}).sort("timestamp", -1)
    documents = await cursor.to_list(length=50)
    
    # MongoDB ObjectIds are not JSON serializable by default
    # So we use bson's json_util to dump it, then parse back to dict
    json_docs = json.loads(json_util.dumps(documents))
    
    # Clean up the object id syntax
    for doc in json_docs:
        doc["_id"] = str(doc["_id"]["$oid"])
        if "timestamp" in doc and "$date" in doc["timestamp"]:
            # Make dates nice
            doc["timestamp"] = doc["timestamp"]["$date"]
            
    return {"history": json_docs}
