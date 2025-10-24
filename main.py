from core.cds_engine import CDSEngine

POSTGRES_CONFIG = {"dbname": "cds_knowledge_base", "user": "cds_user", "password": "cds_password", "host": "localhost",
                   "port": "5433"}
REDIS_CONFIG = {"host": "localhost", "port": 6379, "db": 0}
MODELS_PATH = 'models'


def print_alert(alert, source=""):
    score = f"(ML Score: {alert['ml_score']:.2f})" if 'ml_score' in alert else ""
    print(f"  - {source}[Loại: {alert['type']}] [Mức độ: {alert['severity']}] {score}")
    print(f"    Nội dung: {alert['message']}")


def run_comprehensive_scenario():
    try:
        engine = CDSEngine(REDIS_CONFIG, POSTGRES_CONFIG, MODELS_PATH)
    except Exception as e:
        print(f"Lỗi khởi tạo Engine: {e}")
        return


    patient_profile = {
        "age": 68,
        "gender": "female",
        "allergies": ["C10AA07"],
        "chronic_diseases": ["N18", "K25"],
        "existing_medications": ["J01FA09"]
    }

    prescription = [
        {"atc_code": "B01AA03", "drug_name": "Warfarin", "dose": 5, "unit": "mg/day"},
        {"atc_code": "N02BA01", "drug_name": "Aspirin", "dose": 81, "unit": "mg/day"},
        {"atc_code": "M01AE01", "drug_name": "Ibuprofen", "dose": 400, "unit": "mg/day"},
        {"atc_code": "N02BE01", "drug_name": "Paracetamol", "dose": 5000, "unit": "mg/day"},
        {"atc_code": "C09AA02", "drug_name": "Enalapril", "dose": 10, "unit": "mg/day"},
        {"atc_code": "C10AA07", "drug_name": "Atorvastatin", "dose": 20, "unit": "mg/day"}
    ]

    raw_alerts, filtered_alerts = engine.check_prescription(patient_profile, prescription)

    print("\n--- 1. CẢNH BÁO THÔ (TỪ HỆ THỐNG LUẬT) ---")
    if not raw_alerts:
        print("Không tìm thấy cảnh báo nào.")
    else:
        print(f"Hệ thống luật đã phát hiện {len(raw_alerts)} cảnh báo tiềm ẩn:")
        for alert in raw_alerts:
            print_alert(alert)

    print("\n\n--- 2. CẢNH BÁO CUỐI CÙNG (SAU KHI LỌC BẰNG AI) ---")
    if not filtered_alerts:
        print("Tất cả các cảnh báo không quan trọng đã được AI lọc bỏ.")
    else:
        print(f"AI đề xuất hiển thị {len(filtered_alerts)} cảnh báo quan trọng nhất cho bác sĩ:")
        for alert in filtered_alerts:
            print_alert(alert, "")

if __name__ == "__main__":
    run_comprehensive_scenario()