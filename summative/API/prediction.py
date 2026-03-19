import json
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat, conint, conlist

EducationRequired = Literal["Associate", "Bachelor", "Master", "PhD"]
ExperienceLevel = Literal["EN", "MI", "SE", "EX"]
CompanySize = Literal["S", "M", "L"]
EmploymentType = Literal["CT", "FL", "PT", "FT"]

JobTitle = Literal[
    "AI Research Scientist",
    "AI Software Engineer",
    "AI Specialist",
    "NLP Engineer",
    "AI Consultant",
    "AI Architect",
    "Principal Data Scientist",
    "Data Analyst",
    "Autonomous Systems Engineer",
    "AI Product Manager",
    "Machine Learning Engineer",
    "Data Engineer",
    "Research Scientist",
    "ML Ops Engineer",
    "Robotics Engineer",
    "Head of AI",
    "Deep Learning Engineer",
    "Data Scientist",
    "Machine Learning Researcher",
    "Computer Vision Engineer",
]

Industry = Literal[
    "Automotive",
    "Media",
    "Education",
    "Consulting",
    "Healthcare",
    "Gaming",
    "Government",
    "Telecommunications",
    "Manufacturing",
    "Energy",
    "Technology",
    "Real Estate",
    "Finance",
    "Transportation",
    "Retail",
]

Country = Literal[
    "China",
    "Canada",
    "Switzerland",
    "India",
    "France",
    "Germany",
    "United Kingdom",
    "Singapore",
    "Austria",
    "Sweden",
    "South Korea",
    "Norway",
    "Netherlands",
    "United States",
    "Israel",
    "Australia",
    "Ireland",
    "Denmark",
    "Finland",
    "Japan",
]

Skill = Literal[
    "Python",
    "SQL",
    "TensorFlow",
    "Kubernetes",
    "Scala",
    "PyTorch",
    "Linux",
    "Git",
    "Java",
    "GCP",
    "Hadoop",
    "Tableau",
    "R",
    "Computer Vision",
    "Data Visualization",
    "Deep Learning",
    "MLOps",
    "Spark",
    "NLP",
    "Azure",
    "AWS",
    "Mathematics",
    "Docker",
    "Statistics",
]

education_order: dict[str, int] = {"Associate": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
experience_order: dict[str, int] = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
size_order: dict[str, int] = {"S": 0, "M": 1, "L": 2}

job_category_map: dict[str, str] = {
    "AI Research Scientist": "Research",
    "Research Scientist": "Research",
    "Machine Learning Researcher": "Research",
    "AI Software Engineer": "Engineering",
    "Machine Learning Engineer": "Engineering",
    "Deep Learning Engineer": "Engineering",
    "NLP Engineer": "Engineering",
    "Computer Vision Engineer": "Engineering",
    "Autonomous Systems Engineer": "Engineering",
    "Robotics Engineer": "Engineering",
    "Data Engineer": "Engineering",
    "ML Ops Engineer": "Engineering",
    "Data Scientist": "Data & Analytics",
    "Principal Data Scientist": "Data & Analytics",
    "Data Analyst": "Data & Analytics",
    "Head of AI": "Leadership & Strategy",
    "AI Architect": "Leadership & Strategy",
    "AI Product Manager": "Leadership & Strategy",
    "AI Consultant": "Leadership & Strategy",
    "AI Specialist": "Specialist",
}

region_map: dict[str, str] = {
    "United States": "North America",
    "Canada": "North America",
    "Austria": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Netherlands": "Europe",
    "Norway": "Europe",
    "Sweden": "Europe",
    "Switzerland": "Europe",
    "Ireland": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "India": "Asia",
    "China": "Asia",
    "Japan": "Asia",
    "Singapore": "Asia",
    "South Korea": "Asia",
    "Israel": "Middle East",
    "Australia": "Oceania",
}

job_category_dummies = [
    "Engineering",
    "Leadership & Strategy",
    "Research",
    "Specialist",
]
region_dummies = ["Europe", "Middle East", "North America", "Oceania"]
employment_dummies = ["FL", "FT", "PT"]
industry_dummies = [
    "Consulting",
    "Education",
    "Energy",
    "Finance",
    "Gaming",
    "Government",
    "Healthcare",
    "Manufacturing",
    "Media",
    "Real Estate",
    "Retail",
    "Technology",
    "Telecommunications",
    "Transportation",
]
all_skills = [
    "Python",
    "SQL",
    "TensorFlow",
    "Kubernetes",
    "Scala",
    "PyTorch",
    "Linux",
    "Git",
    "Java",
    "GCP",
    "Hadoop",
    "Tableau",
    "R",
    "Computer Vision",
    "Data Visualization",
    "Deep Learning",
    "MLOps",
    "Spark",
    "NLP",
    "Azure",
    "AWS",
    "Mathematics",
    "Docker",
    "Statistics",
]

base_dir = Path(__file__).resolve().parent
model_dir = (base_dir / ".." / "linear_regression").resolve()
model_path = model_dir / "best_model.pkl"
scaler_path = model_dir / "scaler.pkl"
columns_path = Path(base_dir / "feature_columns.json").resolve()


try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = json.loads(columns_path.read_text())
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


class PredictionRequest(BaseModel):
    job_title: JobTitle
    employment_type: EmploymentType
    industry: Industry
    company_location: Country
    employee_residence: Country
    experience_level: ExperienceLevel
    company_size: CompanySize
    education_required: EducationRequired
    remote_ratio: conint(ge=0, le=100) = Field(..., description="0, 50 or 100")
    years_experience: conint(ge=0, le=50)
    job_description_length: conint(ge=0, le=20000)
    benefits_score: confloat(ge=0.0, le=10.0)
    required_skills: conlist(Skill, min_length=0, max_length=24) = Field(
        default_factory=list
    )


class PredictResponse(BaseModel):
    predicted_salary_usd: float


def encode_request(req: PredictionRequest) -> dict[str, float]:
    job_category = job_category_map[req.job_title]
    company_region = region_map.get(req.company_location, "Other")
    employee_region = region_map.get(req.employee_residence, "Other")
    skills_set = set(req.required_skills)

    row: dict[str, float] = {
        "experience_level": float(experience_order[req.experience_level]),
        "company_size": float(size_order[req.company_size]),
        "remote_ratio": float(req.remote_ratio),
        "education_required": float(education_order[req.education_required]),
        "years_experience": float(req.years_experience),
        "job_description_length": float(req.job_description_length),
        "benefits_score": float(req.benefits_score),
    }

    for c in job_category_dummies:
        row[f"job_category_{c}"] = 1.0 if job_category == c else 0.0

    for c in region_dummies:
        row[f"company_region_{c}"] = 1.0 if company_region == c else 0.0
        row[f"employee_region_{c}"] = 1.0 if employee_region == c else 0.0

    for c in employment_dummies:
        row[f"employment_type_{c}"] = 1.0 if req.employment_type == c else 0.0

    for c in industry_dummies:
        row[f"industry_{c}"] = 1.0 if req.industry == c else 0.0

    for s in all_skills:
        row[f"skill_{s.replace(' ', '_')}"] = 1.0 if s in skills_set else 0.0

    return row


app = FastAPI(
    title="AI Job Market Salary Predictor API",
    version="1.0.0",
    description="Predict AI job salary (USD) from job attributes.",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "AI Job Market Salary Predictor API",
        "swagger_ui": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model": model_path.exists(),
        "scaler": scaler_path.exists(),
        "feature_cols": columns_path.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictionRequest) -> PredictResponse:
    x_row = pd.DataFrame([encode_request(req)]).reindex(
        columns=feature_columns, fill_value=0.0
    )

    try:
        x_scaled = scaler.transform(x_row.to_numpy(dtype=np.float64))
        predicted_salary_usd = float(model.predict(x_scaled)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictResponse(predicted_salary_usd=predicted_salary_usd)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
