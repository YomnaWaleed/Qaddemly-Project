# ATS Scan System - Applicant Tracking System Scanner

## Overview
The ATS Scan System is a FastAPI-based application that analyzes and ranks job applicants based on their similarity to a job description using natural language processing techniques.

## Features
- Extracts and preprocesses resume data from multiple fields
- Processes job descriptions and creates summaries
- Calculates similarity scores using n-gram matching
- Ranks applicants based on their match score
- Provides a REST API endpoint for processing requests

## System Architecture

```mermaid
%%{init: {'theme': 'neutral', 'fontFamily': 'Arial', 'gantt': {'barHeight': 20}}}%%
flowchart TD
    A[Client] -->|POST JSON Data| B[FastAPI Endpoint]
    B --> C[Process Data]
    C --> D[Extract Resume Data]
    C --> E[Extract Job Description]
    D --> F[Preprocess Text]
    E --> F
    F --> G[Summarize Job Description]
    G --> H[Calculate Similarity Scores]
    H --> I[Rank Applicants]
    I --> J[Return Ranked Results]
    J --> K[Client]
```