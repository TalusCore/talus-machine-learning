"""
FastAPI Backend for Health & Fitness Insights ML
Handles ML analysis and serves results via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
from health_fitness_insights import HealthFitnessInsights

# Initialize FastAPI app
app = FastAPI(title="Health Fitness Insights API", version="1.0.0")

# Add CORS middleware to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:3001",
        "https://yourdomain.com",  # Production domain
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML model (done once at startup)
ml_model = None

@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup"""
    import time
    import kagglehub
    global ml_model
    try:
        print("\n" + "="*70)
        print("LOADING ML MODEL - THIS MAY TAKE 2-5 MINUTES ON FIRST RUN")
        print("="*70)

        # Download dataset at runtime (Render build cache doesn't persist to runtime)
        print("\n[STEP 0/3] Downloading dataset from Kaggle...")
        dataset_path = os.getenv('DATASET_PATH')  # Check if manually overridden
        if not dataset_path:
            dataset_path = kagglehub.dataset_download("evan65549/health-and-fitness-dataset")
            os.environ['DATASET_PATH'] = dataset_path
            print(f"   Dataset downloaded to: {dataset_path}")
        else:
            print(f"   Using existing DATASET_PATH: {dataset_path}")

        # Step 1: Load dataset
        start_time = time.time()
        print("\n[STEP 1/3] Loading dataset...")
        ml_model = HealthFitnessInsights(dataset_path)
        load_time = time.time() - start_time
        print(f"   SUCCESS: Dataset loaded in {load_time:.1f}s")
        
        # Step 2: Perform clustering
        start_time = time.time()
        print("\n[STEP 2/3] Performing clustering analysis...")
        ml_model.perform_clustering()
        clustering_time = time.time() - start_time
        print(f"   SUCCESS: Clustering complete in {clustering_time:.1f}s")
        
        # Step 3: Train anomaly detectors
        start_time = time.time()
        print("\n[STEP 3/3] Training anomaly detectors...")
        ml_model.train_anomaly_detector()
        detector_time = time.time() - start_time
        print(f"   SUCCESS: Anomaly detectors trained in {detector_time:.1f}s")
        
        total_time = load_time + clustering_time + detector_time
        print("\n" + "="*70)
        print(f"ML MODEL READY! Total load time: {total_time:.1f}s")
        print("="*70)
        print("\nAPI Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health\n")
        
    except Exception as e:
        print(f"\nERROR: Failed to load ML model: {e}")
        import traceback
        traceback.print_exc()
        raise

# Pydantic models for request/response validation
class UserFitnessData(BaseModel):
    """Input model for user fitness analysis"""
    age: float
    weight_kg: float
    height_cm: float
    steps_per_day: float
    workout_frequency_per_week: float
    heart_rate_resting: float
    exercise_minutes_per_day: float
    gender: Optional[str] = "Unknown"
    fitness_level: Optional[str] = "Moderately Active"
    user_id: Optional[str] = None  # Supabase user ID

class FitnessInsights(BaseModel):
    """Output model for analysis results"""
    user_id: Optional[str] = None
    age_bracket: str
    cluster: int
    cluster_size: int
    cluster_percentage: float
    anomaly_score: float
    is_anomaly: bool
    percentiles: Dict[str, float]
    struggles: List[tuple]
    strengths: List[tuple]
    cluster_deviations: List[tuple]
    recommendations: List[Dict[str, Any]]

# API Routes

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None
    }

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_user(user_data: UserFitnessData):
    """
    Main endpoint: Analyze user fitness data
    
    Args:
        user_data: User's fitness metrics
        
    Returns:
        Fitness insights and recommendations
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Convert Pydantic model to dictionary
        user_dict = user_data.dict()
        
        # Run analysis
        insights = ml_model.analyze_user(user_dict)
        
        # Add user_id if provided
        if user_data.user_id:
            insights['user_id'] = user_data.user_id
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze(users: List[UserFitnessData]):
    """
    Batch analyze multiple users
    
    Args:
        users: List of user fitness data
        
    Returns:
        List of insights for all users
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        results = []
        for user_data in users:
            user_dict = user_data.dict()
            insights = ml_model.analyze_user(user_dict)
            if user_data.user_id:
                insights['user_id'] = user_data.user_id
            results.append(insights)
        
        return {"count": len(results), "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch analysis failed: {str(e)}")

@app.get("/cluster-summary")
async def cluster_summary():
    """
    Get summary of all clusters
    
    Returns:
        Cluster profiles and statistics
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        summary = {}
        for bracket_label, clusters in ml_model.cluster_profiles.items():
            summary[bracket_label] = {}
            for cluster_id, profile in clusters.items():
                summary[bracket_label][cluster_id] = {
                    col: {
                        'mean': float(profile[col]['mean']),
                        'std': float(profile[col]['std']),
                        'median': float(profile[col]['median'])
                    }
                    for col in profile.keys()
                }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get summary: {str(e)}")

@app.get("/metrics")
async def available_metrics():
    """
    Get available metrics and their classifications
    
    Returns:
        Metric names and whether higher/lower is better
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    return {
        "numeric_columns": ml_model.numeric_cols,
        "categorical_columns": ml_model.categorical_cols,
        "higher_is_better": list(ml_model.METRICS_HIGHER_IS_BETTER),
        "lower_is_better": list(ml_model.METRICS_LOWER_IS_BETTER),
        "non_actionable": list(ml_model.NON_ACTIONABLE_COLS)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run with: uvicorn fastapi_ml_backend:app --reload --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )