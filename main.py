"""
FastAPI Backend for Health & Fitness Insights ML
Handles ML analysis and serves results via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import os
from health_fitness_insights import HealthFitnessInsights

app = FastAPI(title="Health Fitness Insights API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_model = None

@app.on_event("startup")
async def startup_event():
    import time
    global ml_model
    try:
        print("\n" + "="*70)
        print("LOADING ML MODEL - THIS MAY TAKE 2-5 MINUTES ON FIRST RUN")
        print("="*70)

        dataset_path = os.getenv('DATASET_PATH')
        if not dataset_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_dir, 'health_fitness_dataset.csv'),
                os.path.join(base_dir, 'data', 'health_fitness_dataset.csv'),
                os.path.join(os.getcwd(), 'health_fitness_dataset.csv'),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    dataset_path = candidate
                    break

        if not dataset_path:
            raise FileNotFoundError(
                "health_fitness_dataset.csv not found. "
                "Place it in the project root or set DATASET_PATH in your .env file."
            )

        print(f"\n[STEP 0/3] Dataset located: {dataset_path}")

        start_time = time.time()
        print("\n[STEP 1/3] Loading dataset...")
        ml_model = HealthFitnessInsights(dataset_path)
        load_time = time.time() - start_time
        print(f"   SUCCESS: Dataset loaded in {load_time:.1f}s")

        start_time = time.time()
        print("\n[STEP 2/3] Performing clustering analysis...")
        ml_model.perform_clustering()
        clustering_time = time.time() - start_time
        print(f"   SUCCESS: Clustering complete in {clustering_time:.1f}s")

        start_time = time.time()
        print("\n[STEP 3/3] Training anomaly detectors...")
        ml_model.train_anomaly_detector()
        detector_time = time.time() - start_time
        print(f"   SUCCESS: Anomaly detectors trained in {detector_time:.1f}s")

        total_time = load_time + clustering_time + detector_time
        print("\n" + "="*70)
        print(f"ML MODEL READY! Total load time: {total_time:.1f}s")
        print("="*70)

    except Exception as e:
        print(f"\nERROR: Failed to load ML model: {e}")
        import traceback
        traceback.print_exc()
        raise


class UserFitnessData(BaseModel):
    """Input model for user fitness analysis.

    fitness_level accepts a 0–100 score:
      0   = completely sedentary / just starting out
      50  = moderately active
      100 = elite / extremely active
    """
    age: float
    weight_kg: float
    height_cm: float
    steps_per_day: float
    workout_frequency_per_week: float
    avg_heart_rate: float
    exercise_minutes_per_day: float
    gender: Optional[str] = "Unknown"
    fitness_level: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Overall fitness score from 0 (sedentary) to 100 (elite)"
    )
    user_id: Optional[str] = None

    @validator('fitness_level')
    def fitness_level_in_range(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('fitness_level must be between 0 and 100')
        return v


class FitnessInsights(BaseModel):
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
    fitness_level_score: Optional[float] = None  # echoes back the 0–100 input


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": ml_model is not None
    }


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_user(user_data: UserFitnessData):
    """Analyze user fitness data and return insights.

    Send fitness_level as a number between 0 and 100.
    The API handles the internal conversion to the dataset scale.
    """
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        user_dict = user_data.dict()
        insights = ml_model.analyze_user(user_dict)
        if user_data.user_id:
            insights['user_id'] = user_data.user_id
        return insights

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze(users: List[UserFitnessData]):
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
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    return {
        "numeric_columns": ml_model.numeric_cols,
        "categorical_columns": ml_model.categorical_cols,
        "higher_is_better": list(ml_model.METRICS_HIGHER_IS_BETTER),
        "lower_is_better": list(ml_model.METRICS_LOWER_IS_BETTER),
        "non_actionable": list(ml_model.NON_ACTIONABLE_COLS),
        "fitness_level_input": "0–100 scale (0=sedentary, 100=elite)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )