import io
import json
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat, conint, conlist
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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


def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["job_id", "salary_currency", "posting_date",
                 "application_deadline", "company_name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df["experience_level"] = df["experience_level"].map(experience_order).astype(float)
    df["company_size"] = df["company_size"].map(size_order).astype(float)
    df["education_required"] = df["education_required"].map(education_order).astype(float)

    df["job_category"] = df["job_title"].map(job_category_map)
    df.drop(columns=["job_title"], inplace=True)
    df = pd.get_dummies(df, columns=["job_category"], drop_first=True, dtype=int)

    df["company_region"] = df["company_location"].map(region_map).fillna("Other")
    df["employee_region"] = df["employee_residence"].map(region_map).fillna("Other")
    df.drop(columns=["company_location", "employee_residence"], inplace=True)
    df = pd.get_dummies(df, columns=["company_region", "employee_region"],
                        drop_first=True, dtype=int)

    df = pd.get_dummies(df, columns=["employment_type", "industry"],
                        drop_first=True, dtype=int)

    if "required_skills" in df.columns:
        for s in all_skills:
            col = f"skill_{s.replace(' ', '_')}"
            df[col] = df["required_skills"].apply(
                lambda val: 1 if isinstance(val, str) and s in val else 0
            )
        df.drop(columns=["required_skills"], inplace=True)

    return df


class RetrainResponse(BaseModel):
    message: str
    r2_score: float
    mse: float
    rows_used: int


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)) -> RetrainResponse:
    global model, scaler, feature_columns

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if "salary_usd" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'salary_usd' target column.",
        )

    try:
        df = _feature_engineer(df)

        y = df["salary_usd"]
        x = df.drop(columns=["salary_usd"])

        new_feature_columns = x.columns.tolist()

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        new_scaler = StandardScaler()
        x_train_scaled = new_scaler.fit_transform(x_train)
        x_test_scaled = new_scaler.transform(x_test)

        param_grid = {
            "max_depth": [10, 20, 30],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [4, 8, 12],
        }
        grid = GridSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(x_train_scaled, y_train)
        new_model = grid.best_estimator_

        y_pred = new_model.predict(x_test_scaled)
        score = float(r2_score(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))

        joblib.dump(new_model, model_path)
        joblib.dump(new_scaler, scaler_path)
        columns_path.write_text(json.dumps(new_feature_columns))

        model = new_model
        scaler = new_scaler
        feature_columns = new_feature_columns

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")

    return RetrainResponse(
        message=f"Model retrained successfully. Best params: {grid.best_params_}",
        r2_score=score,
        mse=mse,
        rows_used=len(df),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
