#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API module for job recommendation system.

This module provides a FastAPI interface to the job recommendation system,
allowing clients to submit user profiles and job data to receive recommendations.
"""

from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from data_preprocessing import DataPreprocessor
from recommendation_to_user import JobRecommender

app = FastAPI(title="Job Recommendation API",
              description="API for recommending jobs to users based on skills, experience, and preferences")


# Pydantic Models
class Skill(BaseModel):
    """Model for user skills."""
    name: str


class Education(BaseModel):
    """Model for user education history."""
    id: Optional[int] = None
    university: str
    field_of_study: str
    gpa: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class Experience(BaseModel):
    """Model for user work experience."""
    id: Optional[int] = None
    job_title: str
    employment_type: str
    company_name: str
    location: str
    location_type: str
    still_working: Optional[bool] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class User(BaseModel):
    """Model for user data."""
    id: int
    country: Optional[str] = None
    city: Optional[str] = None
    about_me: str
    subtitle: str
    skills: List[Dict[str, str]]
    educations: List[Dict[str, Any]]
    experiences: List[Dict[str, Any]]


class Job(BaseModel):
    """Model for job data."""
    id: int
    title: str
    description: str
    country: Optional[str] = None
    city: Optional[str] = None
    location_type: Optional[str] = None
    status: Optional[str] = None
    skills: List[str]
    salary: Optional[Union[int, float]] = None
    employee_type: Optional[str] = None
    keywords: Optional[List[str]]
    experience: Optional[int] = None
    business_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobsRequest(BaseModel):
    """Model for job recommendation request."""
    user: User
    jobs: List[Job]


class InputData(BaseModel):
    """Model for complete input data."""
    success: bool = True
    user: User
    jobs: List[Job]


class RecommendationResponse(BaseModel):
    """Model for recommendation response."""
    recommendations: List[Dict[str, Any]]
    message: str = "Job recommendations generated successfully"


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_jobs(request: InputData):
    """
    Generate job recommendations for a user based on input data.

    Parameters
    ----------
    request : InputData
        The input data containing user profile and jobs

    Returns
    -------
    dict
        Dictionary with recommendations and status message
    """
    try:
        # Convert input data to internal format
        input_data = {
            "user": request.user.model_dump(),
            "jobs": [job.model_dump() for job in request.jobs],
            "select": request.select
        }

        # Preprocess data
        preprocessor = DataPreprocessor()
        user_df, jobs_df, skill_embeddings = preprocessor.preprocess(input_data)

        # Generate recommendations
        recommender = JobRecommender(skill_embeddings= skill_embeddings, top_n=input_data['select'])
        recommendations = recommender.recommend(user_df, jobs_df)

        return {
            "recommendations": recommendations,
            "message": "Job recommendations generated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)