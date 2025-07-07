#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API module for job matching score calculation.

This module provides a FastAPI interface to calculate matching scores between users and jobs,
based on skills, experience, and preferences.
"""
import pickle
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from data_preprocessing import DataPreprocessor
from matching_score import MatchingScore

app = FastAPI(title="Job Matching Score API",
              description="API for calculating matching scores between users and jobs based on skills, experience, and preferences")

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

class InputData(BaseModel):
    """Model for complete input data."""
    success: bool = True
    user: User
    job: Job

class MatchScoreResponse(BaseModel):
    """Model for matching score response."""
    similarity_score: float
    message: str = "Matching score calculated successfully"

@app.post("/match-score", response_model=MatchScoreResponse)
async def calculate_match_score(request: InputData):
    """
    Calculate the matching score between a user and a job based on input data.

    Parameters
    ----------
    request : InputData
        The input data containing user profile and job details

    Returns
    -------
    MatchScoreResponse
        Response with similarity_score and status message
    """
    try:
        # Convert input data to internal format
        input_data = {
            "user": request.user.dict(),
            "job": request.job.dict()
        }

        # Preprocess data
        preprocessor = DataPreprocessor()
        user, job = preprocessor.preprocess(input_data)

        # Load precomputed skill embeddings from local file
        with open('skill_embeddings.pkl', 'rb') as f:
            skill_embeddings = pickle.load(f)

        # Calculate matching score
        scorer = MatchingScore(preprocessor.model, skill_embeddings)
        final_score = scorer.find_score(user, job)

        return MatchScoreResponse(similarity_score=final_score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating matching score: {str(e)}")

if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)