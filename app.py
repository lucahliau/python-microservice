from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import calculate_preferences  # our clustering module
import recommend             # our recommendation module

app = FastAPI(title="Python Recommendation Microservice")

# ----------------------------
# Request models
# ----------------------------

class CalculatePreferencesRequest(BaseModel):
    likedDescriptions: List[str] = []
    dislikedDescriptions: List[str] = []

class RecommendRequest(BaseModel):
    likedClusters: List[List[float]]
    dislikedClusters: List[List[float]] = []
    # Each post should include at least an 'id', an 'embedding', and optionally a 'description'
    posts: List[dict]

# ----------------------------
# Endpoints
# ----------------------------

@app.post("/calculate_preferences")
async def calculate_preferences_endpoint(request: CalculatePreferencesRequest):
    try:
        # Call the helper function from calculate_preferences.py
        result = calculate_preferences.calculate_preferences(
            liked_descriptions=request.likedDescriptions,
            disliked_descriptions=request.dislikedDescriptions
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
async def recommend_endpoint(request: RecommendRequest):
    try:
        # Call the helper function from recommend.py
        recommendations = recommend.get_recommendations(
            liked_clusters=request.likedClusters,
            disliked_clusters=request.dislikedClusters,
            posts=request.posts
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
