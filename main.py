import json
import pandas as pd
from sqlalchemy import create_engine, text
from core.cds_engine import CDSEngine

POSTGRES_CONFIG = {"dbname": "cds_knowledge_base", "user": "cds_user", "password": "cds_password", "host": "localhost",
                   "port": "5433"}
REDIS_CONFIG = {"host": "localhost", "port": 6379, "db": 0}
MODELS_PATH = 'models'

db_engine = create_engine(
    f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['dbname']}")


def print_alert(alert, icon=""):
    ml_info = f"[AI Score: {alert.get('ml_score', 0):.2f}]" if 'ml_score' in alert else ""
    rule_id = alert.get('rule_id', 'N/A')
    print(f" {icon} Loại: {alert['type']:<10} | Mức độ: {alert['severity']:<10}")
    print(f"    Nội dung: {alert['message']}")
    # print(f"    🔍 Rule ID: {rule_id}")
    print("-" * 60)


def run_mega_test():
    print("\n" + "█" * 80)
    print("🏥 KỊCH BẢN TEST: ĐƠN THUỐC 'THẢM HỌA' (8 LOẠI THUỐC)")
    print("█" * 80 + "\n")

    try:
        cds = CDSEngine(REDIS_CONFIG, POSTGRES_CONFIG, MODELS_PATH)
        print("✅ CDS Engine đã sẵn sàng.\n")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    patient = {
        "age": 82,
        "gender": "Male",
        "allergies": ["J01"],
        "chronic_diseases": ["1D65", "1F0Y", "1C60"],
        "existing_medications": []
    }

    prescription = [
        {"atc_code": "J01CA04", "drug_name": "Amoxicillin", "dose": 500, "unit": "mg"},
        {"atc_code": "N05BA12", "drug_name": "ALPRAZOLAM", "dose": 500, "unit": "mg"},
        {"atc_code": "N02BE01", "drug_name": "Paracetamol", "dose": 10000, "unit": "mg"},
        {"atc_code": "S01EC01", "drug_name": "ACETAZOLAMIDE", "dose": 10000, "unit": "mg"},

        {"atc_code": "C07AA05", "drug_name": "Propranolol", "dose": 40, "unit": "mg"},
        {"atc_code": "J01GB03", "drug_name": "Gentamicin", "dose": 80, "unit": "mg"},
        {"atc_code": "M01AE01", "drug_name": "Ibuprofen", "dose": 400, "unit": "mg"},

        {"atc_code": "B01AA03", "drug_name": "Warfarin", "dose": 5, "unit": "mg"},
        {"atc_code": "C03DA01", "drug_name": "Spironolactone", "dose": 25, "unit": "mg"},

        {"atc_code": "A11GA01", "drug_name": "Vitamin C", "dose": 500, "unit": "mg"}
    ]

    print(f" Bệnh nhân: 82 tuổi. Bệnh nền: Suy thận (N18), Dạ dày (K25), Hen (J45). Dị ứng: J01.")
    print(f" Danh sách thuốc: {', '.join([d['drug_name'] for d in prescription])}")

    raw_alerts, filtered_alerts = cds.check_prescription(patient, prescription)

    filtered_ids = {id(a) for a in filtered_alerts}

    print(f"\n🔴 [BÁC SĨ CẦN CHÚ Ý] - AI Đề xuất hiển thị ({len(filtered_alerts)} cảnh báo):")
    if filtered_alerts:
        for a in filtered_alerts:
            icon = "☠️" if a['type'] in ['ALLERGY', 'DOSE'] else "🚫"
            if a['type'] == 'DDI': icon = "⚡"
            print_alert(a, icon)
    else:
        print("   Không có cảnh báo quan trọng.")

    # Tìm những cảnh báo đã bị AI lọc bỏ (Có trong Raw nhưng không có trong Filtered)
    # Logic so sánh đơn giản dựa trên message
    filtered_msgs = [a['message'] for a in filtered_alerts]
    dropped_alerts = [a for a in raw_alerts if a['message'] not in filtered_msgs]

    print(f"\n🟢 [ĐÃ ĐƯỢC AI LỌC BỎ] - Coi là nhiễu ({len(dropped_alerts)} cảnh báo):")
    if dropped_alerts:
        for a in dropped_alerts:
            print_alert(a, "🗑️")
    else:
        print("   AI không lọc bỏ cảnh báo nào (Tất cả đều được đánh giá là quan trọng).")

    check_ddxi = any(a['type'] == 'DDxI' for a in raw_alerts)
    if not check_ddxi:
        print("\n⚠️ [DEBUG] Không thấy cảnh báo DDxI nào xuất hiện!")
        print("   -> Đang kiểm tra Database xem có Rule cho Propranolol (C07AA05) không...")
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM ddxi_rules WHERE drug_atc = 'C07AA05' LIMIT 3")).fetchall()
            if result:
                print(f"   -> Có dữ liệu trong DB: {result}")
                print("   -> Có thể do mã ICD bệnh nhân không khớp chính xác với mã trong Rule.")
            else:
                print("   -> ❌ Không tìm thấy Rule nào trong bảng ddxi_rules cho C07AA05.")
                print("   -> Bạn cần chạy lại file ETL để nạp dữ liệu kỹ hơn.")


if __name__ == "__main__":
    run_mega_test()