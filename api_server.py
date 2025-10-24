from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from core.cds_engine import CDSEngine

class Drug(BaseModel):
    atc_code: str
    drug_name: str
    dose: Optional[float] = None
    unit: Optional[str] = None


class PatientProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    allergies: List[str] = Field(default_factory=list)
    chronic_diseases: List[str] = Field(default_factory=list)
    existing_medications: List[str] = Field(default_factory=list)


class CDSRequest(BaseModel):
    patient_profile: PatientProfile
    prescription: List[Drug]


app = FastAPI(
    title="Clinical Decision Support API",
    description="Một API để cung cấp cảnh báo y khoa thông minh dựa trên luật và AI.",
    version="1.0.0"
)


POSTGRES_CONFIG = {
    "dbname": "cds_knowledge_base",
    "user": "cds_user",
    "password": "cds_password",
    "host": "localhost",
    "port": "5433"
}
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}
MODELS_PATH = 'models'

try:
    cds_engine = CDSEngine(REDIS_CONFIG, POSTGRES_CONFIG, MODELS_PATH)
except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo CDSEngine. API sẽ không hoạt động. Lỗi: {e}")
    cds_engine = None


@app.post("/check-prescription/", tags=["CDS"])
def check_prescription_api(request: CDSRequest):

    if not cds_engine:
        raise HTTPException(status_code=500, detail="Lỗi: Dịch vụ CDS không khả dụng do lỗi khởi tạo.")

    patient_profile_dict = request.patient_profile.dict()
    prescription_list = [drug.dict() for drug in request.prescription]

    raw_alerts, filtered_alerts = cds_engine.check_prescription(
        patient_profile=patient_profile_dict,
        prescription=prescription_list
    )

    return {
        "raw_alerts_count": len(raw_alerts),
        "filtered_alerts_count": len(filtered_alerts),
        "final_alerts": filtered_alerts
    }


@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "Clinical Decision Support API is running!"}