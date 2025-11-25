import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

# --- CẤU HÌNH ---
DB_URL = "postgresql://cds_user:cds_password@localhost:5433/cds_knowledge_base"
NUM_RECORDS = 10000  # Sinh 10,000 dòng để train cho tốt
BATCH_SIZE = 2000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
fake = Faker()


def get_db_engine():
    return create_engine(DB_URL)


def fetch_reference_data(engine):
    """Lấy dữ liệu thật từ các bảng Rules để giả lập cho khớp thực tế"""
    with engine.connect() as conn:
        # Lấy danh sách thuốc và bệnh
        drugs = pd.read_sql("SELECT distinct drug_atc FROM dose_rules LIMIT 1000", conn)['drug_atc'].tolist()
        diseases = pd.read_sql("SELECT distinct icd_code FROM ddxi_rules LIMIT 500", conn)['icd_code'].tolist()

        # Lấy cặp DDI thật
        ddi_pairs = pd.read_sql("SELECT drug_a_atc, drug_b_atc, severity FROM ddi_rules", conn).to_dict('records')

        # Lấy cặp DDxI thật
        ddxi_pairs = pd.read_sql("SELECT drug_atc, icd_code, contraindication_level FROM ddxi_rules", conn).to_dict(
            'records')

    return drugs, diseases, ddi_pairs, ddxi_pairs


def format_pg_array(data_list):
    """Chuyển Python List thành Postgres Array String: {item1,item2}"""
    if not data_list:
        return "{}"
    # Loại bỏ các ký tự lạ nếu có và tạo chuỗi định dạng {a,b,c}
    clean_list = [str(x).replace(",", "").strip() for x in data_list]
    return f"{{{','.join(clean_list)}}}"


def generate_row(drugs, diseases, ddi_pairs, ddxi_pairs):
    # Xác suất kịch bản:
    # 50% Bình thường, 30% DDI, 20% DDxI
    scenario = random.choices(['NORMAL', 'DDI', 'DDxI'], weights=[50, 30, 20], k=1)[0]

    # Dữ liệu cơ bản
    patient_age = random.randint(5, 90)
    patient_gender = random.choice(['Male', 'Female'])
    created_at = fake.date_time_between(start_date='-60d', end_date='now')

    list_atc = []
    icd_list = []
    alert_type = 'NONE'
    severity = 'None'
    rule_id = None
    doctor_action = 'NONE'
    final_rx_outcome = 'DISPENSED'

    if scenario == 'NORMAL':
        # Random 2-3 thuốc không xung đột (giả định)
        list_atc = random.sample(drugs, k=random.randint(1, 3))
        icd_list = random.sample(diseases, k=1)

        # Hành động mặc định
        doctor_action = 'NONE'
        final_rx_outcome = 'DISPENSED'

    elif scenario == 'DDI':
        if not ddi_pairs: return None
        pair = random.choice(ddi_pairs)

        list_atc = [pair['drug_a_atc'], pair['drug_b_atc']]
        icd_list = random.sample(diseases, k=1)

        alert_type = 'DDI'
        severity = pair['severity']  # Major, Moderate, Minor...
        rule_id = f"DDI_{pair['drug_a_atc']}_{pair['drug_b_atc']}"

        # --- LOGIC GIẢ LẬP HÀNH VI BÁC SĨ (LABEL) ---
        if severity == 'Major':
            # Nghiêm trọng -> 85% Hủy, 15% Bỏ qua (nếu cấp bách)
            doctor_action = random.choices(['CANCEL', 'OVERRIDE'], weights=[85, 15], k=1)[0]
        elif severity == 'Moderate':
            # Trung bình -> 40% Hủy, 60% Bỏ qua
            doctor_action = random.choices(['CANCEL', 'OVERRIDE'], weights=[40, 60], k=1)[0]
        else:
            # Nhẹ -> 95% Bỏ qua
            doctor_action = 'OVERRIDE'

        final_rx_outcome = 'ABORTED' if doctor_action == 'CANCEL' else 'DISPENSED'

    elif scenario == 'DDxI':
        if not ddxi_pairs: return None
        pair = random.choice(ddxi_pairs)

        list_atc = [pair['drug_atc']]
        # Thêm thuốc rác để list tự nhiên hơn
        if len(drugs) > 5: list_atc.append(random.choice(drugs))

        icd_list = [pair['icd_code']]

        alert_type = 'DDxI'
        severity = pair['contraindication_level']  # Absolute, Relative
        rule_id = f"DDxI_{pair['drug_atc']}_{pair['icd_code']}"

        # --- LOGIC GIẢ LẬP HÀNH VI BÁC SĨ ---
        if severity == 'Absolute':
            # Tuyệt đối -> 90% Hủy
            doctor_action = random.choices(['CANCEL', 'OVERRIDE'], weights=[90, 10], k=1)[0]
        else:
            # Tương đối -> 50/50
            doctor_action = random.choices(['CANCEL', 'OVERRIDE'], weights=[50, 50], k=1)[0]

        final_rx_outcome = 'ABORTED' if doctor_action == 'CANCEL' else 'DISPENSED'

    return {
        "event_id": str(uuid.uuid4()),
        "list_atc": format_pg_array(list_atc),  # Format {A,B} cho Postgres
        "icd_list": format_pg_array(icd_list),
        "alert_type": alert_type,
        "severity": severity,
        "rule_id": rule_id,
        "doctor_action": doctor_action,
        "final_rx_outcome": final_rx_outcome,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "created_at": created_at
    }


def main():
    engine = get_db_engine()
    logging.info("🛠 Đang lấy dữ liệu tham chiếu (Drugs, ICD, Rules)...")

    try:
        drugs, diseases, ddi_pairs, ddxi_pairs = fetch_reference_data(engine)
        if not drugs: raise Exception("Data trống")
    except Exception as e:
        logging.error(f"❌ Lỗi: {e}. Hãy chạy ETL_data.py trước!")
        return

    logging.info(f"🚀 Bắt đầu sinh {NUM_RECORDS} dòng log theo cấu trúc cũ...")

    buffer = []
    total_inserted = 0

    # Xóa dữ liệu cũ nếu muốn train sạch (Optional)
    # with engine.connect() as conn:
    #     conn.execute(text("TRUNCATE TABLE event_logs;"))
    #     conn.commit()

    for i in range(NUM_RECORDS):
        row = generate_row(drugs, diseases, ddi_pairs, ddxi_pairs)
        if row:
            buffer.append(row)

        if len(buffer) >= BATCH_SIZE:
            df = pd.DataFrame(buffer)
            df.to_sql('event_logs', engine, if_exists='append', index=False)
            total_inserted += len(buffer)
            logging.info(f"   -> Đã ghi {total_inserted} dòng...")
            buffer = []

    if buffer:
        df = pd.DataFrame(buffer)
        df.to_sql('event_logs', engine, if_exists='append', index=False)
        logging.info(f"   -> Đã ghi phần còn lại.")

    logging.info("✅ HOÀN TẤT! Dữ liệu đã sẵn sàng.")


if __name__ == "__main__":
    main()