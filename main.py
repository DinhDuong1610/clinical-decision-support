from core.cds_engine import CDSEngine
import time

POSTGRES_CONFIG = {
    "dbname": "cds_knowledge_base",
    "user": "cds_user",
    "password": "cds_password",
    "host": "localhost", "port": "5433"
}
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

def test():
    try:
        engine = CDSEngine(REDIS_CONFIG, POSTGRES_CONFIG)
    except Exception as e:
        print(f"Lỗi khởi tạo Engine: {e}")
        return

    patient_profile = {
        "age": 45, "gender": "male", "allergies": ["M01AE01"],
        "chronic_diseases": ["N18", "K25"], "existing_medications": ["J01FA09"]
    }
    prescription = [
        {"atc_code": "B01AA03", "drug_name": "Warfarin", "dose": 5, "unit": "mg/day"},
        {"atc_code": "N02BA01", "drug_name": "Aspirin", "dose": 325, "unit": "mg/day"},
        {"atc_code": "M01AE01", "drug_name": "Ibuprofen", "dose": 800, "unit": "mg/day"},
        {"atc_code": "N02BE01", "drug_name": "Paracetamol", "dose": 5000, "unit": "mg/day"}
    ]

    alerts = engine.check_prescription(patient_profile, prescription)

    if not alerts:
        print("Không tìm thấy cảnh báo nào.")
    else:
        print(f"Tìm thấy {len(alerts)} cảnh báo:")
        severity_order = {"Major": 0, "Absolute": 1, "Relative": 2, "Moderate": 3, "Unknown": 4}
        sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(x['severity'], 99))

        for i, alert in enumerate(sorted_alerts, 1):
            print(f"\n{i}. CẢNH BÁO [{alert['type']}] [Mức độ: {alert['severity']}]")
            print(f"  {alert['message']}")

    time.sleep(1)

    for alert in alerts:
        doctor_action = "accepted" if alert['severity'] in ["Major", "Absolute"] else "ignored"
        final_outcome = "changed" if doctor_action == "accepted" else "unchanged"

        log_data = {
            "list_atc": [drug['atc_code'] for drug in prescription],
            "icd_list": patient_profile['chronic_diseases'],
            "alert": alert,
            "doctor_action": doctor_action,
            "final_rx_outcome": final_outcome,
            "patient_age": patient_profile['age'],
            "patient_gender": patient_profile['gender']
        }

        engine.log_event_async(log_data)

if __name__ == "__main__":
    test()