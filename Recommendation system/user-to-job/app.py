from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from data_preprocessing import DataPreprocessor
from recommendation_to_job import UserRecommender

app = FastAPI()

class RecommendationRequest(BaseModel):
    """
    Request model for recommending users for a job.

    Attributes
    ----------
    users : List[Dict]
        List of user data dictionaries containing profile information (id, skills, about_me, etc.).
    job : Dict
        Job data dictionary containing job details (id, title, description, skills, etc.).
    """
    users: List[Dict]
    job: Dict

@app.post("/recommend-users")
async def recommend_users(request: RecommendationRequest):
    """
    Recommend users for a given job based on similarity metrics.

    Parameters
    ----------
    request : RecommendationRequest
        Request body containing a list of users and a single job.

    Returns
    -------
    list
        List of dictionaries with user IDs and similarity scores
        (e.g., [{"id": 2, "similarity_score": 0.3}]).

    Raises
    ------
    HTTPException
        400 if input validation fails (e.g., empty users list or missing job data).
        500 if preprocessing or recommendation fails.
    """
    try:
        # Validate input
        if not request.users:
            raise ValueError("Users list cannot be empty")
        if not request.job:
            raise ValueError("Job data is required")

        # Preprocess user and job data
        preprocessor = DataPreprocessor()
        data = {"users": request.users, "job": request.job}
        user_df, jobs_df = preprocessor.preprocess(data)

        # Generate recommendations for the job
        recommender = UserRecommender()
        recommendations = recommender.recommend(jobs_df, user_df)

        return recommendations

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)