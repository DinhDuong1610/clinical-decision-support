import pandas as pd
from sqlalchemy import create_engine, text
import logging
import os
import re
import random  # <--- THÊM: Để random mức độ
from rapidfuzz import process, fuzz

# --- 1. CẤU HÌNH KẾT NỐI VÀ ĐƯỜNG DẪN ---
DB_URL = "postgresql://cds_user:cds_password@localhost:5433/cds_knowledge_base"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATC_FILE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'atc_data.csv')
ICD_FILE_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'icd_data.csv')

# --- 2. CẤU HÌNH TÊN CỘT ---
COL_ICD_CODE = 'ICD11_Code'
COL_ICD_NAME_EN = 'ICD11_Title_EN'
COL_ICD_NAME_VI = 'ICD11_Title_EN_VN'

COL_ATC_CODE = 'Mã ATC'
COL_ATC_NAME = 'Tên chung quốc tế'
COL_ATC_DOSE = 'Dạng thuốc và hàm lượng'
COL_ATC_INTERACTION = 'Tương tác thuốc'
COL_ATC_CONTRA = 'Chống chỉ định'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_db_engine():
    return create_engine(DB_URL)


def read_csv_robust(filepath):
    if not os.path.exists(filepath):
        logging.error(f"❌ Không tìm thấy file: {filepath}")
        return None
    separators = ['\t', ',', ';', '|']
    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip', nrows=5, encoding='utf-8')
            if df.shape[1] > 1:
                logging.info(f"   -> Đọc file bằng dấu phân cách: '{sep}'")
                return pd.read_csv(filepath, sep=sep, on_bad_lines='skip', encoding='utf-8')
        except:
            continue
    try:
        return pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        logging.error(f"❌ Lỗi đọc file: {e}")
        return None


def extract_dose_info(text_content):
    if pd.isna(text_content): return None, None, None
    text_content = str(text_content).lower()
    pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|iu|%)'
    match = re.search(pattern, text_content)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        route = 'oral'
        if any(x in text_content for x in ['tiêm', 'truyền', 'ống']):
            route = 'injection'
        elif any(x in text_content for x in ['bôi', 'ngoài da', 'kem']):
            route = 'topical'
        elif any(x in text_content for x in ['nhỏ mắt', 'tra mắt']):
            route = 'ocular'
        elif any(x in text_content for x in ['đặt', 'âm đạo', 'hậu môn']):
            route = 'rectal/vaginal'
        return value, unit, route
    return None, None, 'oral'


def split_contraindications(text):
    if pd.isna(text): return []
    text = str(text).lower()
    text = text.replace('/', ';')
    parts = re.split(r'[;.,]', text)
    cleaned_parts = []
    ignore_words = ['tiền sử', 'chứng', 'bệnh', 'người bị', 'bệnh nhân']
    for part in parts:
        part = part.strip()
        if len(part) < 4: continue
        for word in ignore_words:
            if part.startswith(word + ' '):
                part = part.replace(word + ' ', '', 1)
        cleaned_parts.append(part)
    return cleaned_parts


# --- HÀM MỚI: TRÍCH XUẤT NỘI DUNG TRONG NGOẶC ---
def extract_mechanism_from_parentheses(text):
    """
    Input: "tetracycline, sắt (giảm hấp thu)"
    Output: "giảm hấp thu"
    """
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1).strip()
    return text.strip()  # Fallback nếu không có ngoặc


def run_etl_process():
    engine = get_db_engine()
    logging.info("🚀 BẮT ĐẦU QUY TRÌNH ETL (TỐI ƯU HÓA LOGIC & RANDOM)")

    # --- BƯỚC 1: ĐỌC DỮ LIỆU ---
    df_icd = read_csv_robust(ICD_FILE_PATH)
    df_atc = read_csv_robust(ATC_FILE_PATH)
    if df_icd is None or df_atc is None: return

    # --- BƯỚC 2: CHUẨN BỊ TỪ ĐIỂN ---
    logging.info("🔄 2. Xây dựng Index...")

    disease_map = {}
    if COL_ICD_CODE in df_icd.columns:
        for _, row in df_icd.iterrows():
            code = str(row[COL_ICD_CODE]).strip()
            # Bỏ qua nếu mã code là nan
            if code.lower() == 'nan': continue

            if COL_ICD_NAME_VI in df_icd.columns and pd.notna(row[COL_ICD_NAME_VI]):
                name_vi = str(row[COL_ICD_NAME_VI]).lower().strip()
                if len(name_vi) > 3: disease_map[name_vi] = code

    disease_names_list = list(disease_map.keys())

    drug_map = {}
    if COL_ATC_CODE in df_atc.columns and COL_ATC_NAME in df_atc.columns:
        for _, row in df_atc.iterrows():
            code = str(row[COL_ATC_CODE]).strip()
            name = str(row[COL_ATC_NAME]).lower().strip()
            if len(code) >= 5 and len(name) > 3:
                drug_map[name] = code

    # --- BƯỚC 3: KHAI PHÁ DỮ LIỆU ---
    logging.info("🔄 3. Đang xử lý chi tiết (Dose, DDI, DDxI)...")

    dose_rules = []
    ddi_rules = []
    ddxi_rules = []

    total_rows = len(df_atc)

    for idx, row in df_atc.iterrows():
        if idx % 200 == 0: print(f"   ... Xử lý {idx}/{total_rows} dòng")

        atc_code = str(row[COL_ATC_CODE]).strip()
        if len(atc_code) < 5: continue

        # A. DOSE RULES
        dose_text = row[COL_ATC_DOSE]
        val, unit, route = extract_dose_info(dose_text)
        if val:
            dose_rules.append({
                'drug_atc': atc_code, 'population': 'adult', 'route': route,
                'dose_min': val * 0.5, 'dose_max': val * 2.0,
                'dose_unit': unit, 'note': str(dose_text)[:100]
            })

        # B. DDI RULES (CẬP NHẬT LOGIC)
        inter_text = str(row[COL_ATC_INTERACTION]).lower()
        if inter_text != 'nan' and len(inter_text) > 5:
            # 1. Tách chuỗi tương tác thành các mệnh đề riêng biệt bằng dấu chấm phẩy
            # Ví dụ: "A, B (tăng độc); C (giảm hiệu quả)" -> ["A, B (tăng độc)", "C (giảm hiệu quả)"]
            clauses = inter_text.split(';')

            for drug_name, other_atc in drug_map.items():
                if atc_code == other_atc: continue

                # Kiểm tra xem tên thuốc B có nằm trong mệnh đề nào không
                found_mechanism = None
                for clause in clauses:
                    if drug_name in clause:
                        # Nếu tìm thấy, trích xuất nội dung trong ngoặc đơn của mệnh đề đó
                        found_mechanism = extract_mechanism_from_parentheses(clause)
                        break  # Đã tìm thấy cơ chế cho thuốc này, thoát vòng lặp clauses

                if found_mechanism:
                    # Random mức độ tương tác
                    rd_severity = random.choice(['Major', 'Moderate', 'Minor'])

                    ddi_rules.append({
                        'drug_a_atc': atc_code,
                        'drug_b_atc': other_atc,
                        'severity': rd_severity,
                        'mechanism': found_mechanism[:255]  # Cắt ngắn nếu quá dài
                    })

        # C. DDxI RULES (CẬP NHẬT LOGIC)
        contra_text = str(row[COL_ATC_CONTRA]).lower()
        if contra_text != 'nan' and len(contra_text) > 3:
            phrases = split_contraindications(contra_text)

            for phrase in phrases:
                if any(x in phrase for x in ['thai', 'bú', 'trẻ', 'mẫn']):
                    continue

                match = process.extractOne(phrase, disease_names_list, scorer=fuzz.token_set_ratio, score_cutoff=70)

                if match:
                    matched_name, score, _ = match
                    icd_code = disease_map.get(matched_name)

                    # Kiểm tra kỹ icd_code không được là nan
                    if icd_code and str(icd_code).lower() != 'nan':
                        # Random mức độ chống chỉ định
                        rd_level = random.choice(['Absolute', 'Relative'])

                        ddxi_rules.append({
                            'drug_atc': atc_code,
                            'icd_code': icd_code,
                            'contraindication_level': rd_level,
                            'statement': f"Gốc: '{phrase}' -> Map: '{matched_name}'"
                        })

    # --- BƯỚC 4: NẠP DATABASE ---
    logging.info(f"📊 Kết quả trích xuất: {len(dose_rules)} Dose, {len(ddi_rules)} DDI, {len(ddxi_rules)} DDxI.")

    try:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE dose_rules, ddi_rules, ddxi_rules RESTART IDENTITY CASCADE;"))
            conn.commit()

        if dose_rules:
            pd.DataFrame(dose_rules).drop_duplicates(subset=['drug_atc']).to_sql('dose_rules', engine,
                                                                                 if_exists='append', index=False)

        if ddi_rules:
            df_ddi = pd.DataFrame(ddi_rules)
            # Xóa trùng lặp cặp (A, B) và (B, A)
            df_ddi['key'] = df_ddi.apply(lambda x: tuple(sorted([x['drug_a_atc'], x['drug_b_atc']])), axis=1)
            df_ddi = df_ddi.drop_duplicates(subset=['key']).drop('key', axis=1)
            df_ddi.to_sql('ddi_rules', engine, if_exists='append', index=False)

        if ddxi_rules:
            df_ddxi = pd.DataFrame(ddxi_rules).drop_duplicates(subset=['drug_atc', 'icd_code'])
            df_ddxi.to_sql('ddxi_rules', engine, if_exists='append', index=False)

        logging.info("✅ Đã nạp dữ liệu vào Database thành công!")

    except Exception as e:
        logging.error(f"❌ Lỗi Database: {e}")


if __name__ == '__main__':
    run_etl_process()