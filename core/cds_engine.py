import redis
import json
import psycopg2
import threading
from itertools import combinations, product
import joblib
import os
import pandas as pd

ALERT_TYPE_DDI = "DDI"
ALERT_TYPE_DDXI = "DDxI"
ALERT_TYPE_ALLERGY = "ALLERGY"
ALERT_TYPE_DOSE = "DOSE"

SEVERITY_MAJOR = "Major"
SEVERITY_ABSOLUTE = "Absolute"
SEVERITY_RELATIVE = "Relative"
SEVERITY_MODERATE = "Moderate"
SEVERITY_UNKNOWN = "Unknown"

class CDSEngine:
    def __init__(self, redis_config, postgres_config, models_path):
        try:
            self.redis_conn = redis.Redis(**redis_config, decode_responses=True)
            self.redis_conn.ping()
            print("DSEngine: Kết nối Redis thành công!")
        except redis.exceptions.ConnectionError as e:
            print(f"CDSEngine: Lỗi kết nối Redis: {e}")
            raise
        self.postgres_config = postgres_config

        self.ml_enabled = self._load_ml_models(models_path)
        if self.ml_enabled:
            print("CDSEngine: Tải mô hình ML thành công! Bộ lọc thông minh đã được kích hoạt.")
        else:
            print("CDSEngine: Không thể tải mô hình ML. Hệ thống sẽ hoạt động ở chế độ chỉ dựa trên luật.")

    def _load_ml_models(self, models_path):
        try:
            self.mlb_atc = joblib.load(os.path.join(models_path, 'mlb_atc.joblib'))
            self.mlb_icd = joblib.load(os.path.join(models_path, 'mlb_icd.joblib'))
            self.column_transformer = joblib.load(os.path.join(models_path, 'column_transformer.joblib'))
            self.model = joblib.load(os.path.join(models_path, 'xgb_model_tuned.joblib'))

            self.feature_names = joblib.load(os.path.join(models_path, 'feature_names.joblib'))

            return True
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file mô hình. {e}")
            return False

    def check_prescription(self, patient_profile, prescription):
        raw_alerts = self._get_raw_alerts(patient_profile, prescription)
        if self.ml_enabled:
            filtered_alerts = self._filter_alerts_with_ml(raw_alerts, patient_profile, prescription)
            return raw_alerts, filtered_alerts
        return raw_alerts, raw_alerts

    def _get_raw_alerts(self, patient_profile, prescription):
        alerts = []
        new_prescription_atcs = {drug['atc_code'] for drug in prescription}
        existing_meds_atcs = set(patient_profile.get('existing_medications', []))
        alerts.extend(self._check_allergies(new_prescription_atcs, patient_profile.get('allergies', [])))
        alerts.extend(self._check_ddxi(new_prescription_atcs, patient_profile.get('chronic_diseases', [])))
        alerts.extend(self._check_ddi(new_prescription_atcs, existing_meds_atcs))
        alerts.extend(self._check_dose(prescription, patient_profile))
        return alerts

    def _filter_alerts_with_ml(self, raw_alerts, patient_profile, prescription):
        final_alerts = []
        context_df = self._prepare_inference_data(patient_profile, prescription)
        for alert in raw_alerts:
            if alert['severity'] in [SEVERITY_MAJOR, SEVERITY_ABSOLUTE]:
                final_alerts.append(alert)
                continue

            inference_df = context_df.copy()
            inference_df['alert_type'] = alert['type']
            inference_df['severity'] = alert['severity']

            feature_vector = self._transform_single_case(inference_df)
            probability_useful = self.model.predict_proba(feature_vector)[0][1]

            threshold = 0.5
            if alert['severity'] in ["Moderate", "Relative"]:
                threshold = 0.4

            if probability_useful >= threshold:
                alert['ml_score'] = round(probability_useful, 2)
                final_alerts.append(alert)
        return final_alerts

    def _prepare_inference_data(self, patient_profile, prescription):
        atc_list = [drug['atc_code'] for drug in prescription]
        data = {'list_atc': [atc_list], 'icd_list': [patient_profile.get('chronic_diseases', [])],
                'patient_age': patient_profile.get('age'), 'patient_gender': patient_profile.get('gender')}
        return pd.DataFrame(data)

    def _transform_single_case(self, df):
        atc_features = self.mlb_atc.transform(df['list_atc'])
        atc_df = pd.DataFrame(atc_features, columns=[f"atc_{c}" for c in self.mlb_atc.classes_])
        icd_features = self.mlb_icd.transform(df['icd_list'])
        icd_df = pd.DataFrame(icd_features, columns=[f"icd_{c}" for c in self.mlb_icd.classes_])
        processed_features = self.column_transformer.transform(df)
        processed_df = pd.DataFrame(processed_features, columns=self.column_transformer.get_feature_names_out())
        combined_df = pd.concat([processed_df, atc_df, icd_df], axis=1)
        return combined_df.reindex(columns=self.feature_names, fill_value=0)

    def log_event_async(self, log_data):
        log_thread = threading.Thread(target=self._write_log_to_db, args=(log_data,))
        log_thread.start()

    def _write_log_to_db(self, log_data):
        sql = """
            INSERT INTO event_logs (
                list_atc, icd_list, alert_type, severity, rule_id, 
                doctor_action, final_rx_outcome, patient_age, patient_gender
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        try:
            with psycopg2.connect(**self.postgres_config) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        log_data['list_atc'],
                        log_data['icd_list'],
                        log_data['alert']['type'],
                        log_data['alert']['severity'],
                        log_data['alert'].get('rule_id', None),
                        log_data['doctor_action'],
                        log_data['final_rx_outcome'],
                        log_data['patient_age'],
                        log_data.get('patient_gender')
                    ))
        except Exception as e:
            print(f"Lỗi khi ghi log: {e}")

    def _check_allergies(self, prescription_atcs, patient_allergies):
        alerts = []
        allergic_drugs = prescription_atcs.intersection(set(patient_allergies))
        for drug_atc in allergic_drugs:
            alerts.append({
                "type": ALERT_TYPE_ALLERGY,
                "severity": SEVERITY_MAJOR,
                "message": f"Bệnh nhân có tiền sử dị ứng với thuốc có mã ATC: {drug_atc}."
            })
        return alerts

    def _check_ddxi(self, prescription_atcs, patient_icds):
        alerts = []
        patient_icds_set = set(patient_icds)
        if not prescription_atcs or not patient_icds_set:
            return []

        pipe = self.redis_conn.pipeline()
        for drug_atc in prescription_atcs:
            pipe.hgetall(f"ddxi:{drug_atc}")
        results = pipe.execute()

        for drug_atc, contraindications in zip(prescription_atcs, results):
            if not contraindications:
                continue
            for icd_code, details_json in contraindications.items():
                if icd_code in patient_icds_set:
                    try:
                        details = json.loads(details_json)
                        alerts.append({
                            "type": ALERT_TYPE_DDXI,
                            "severity": details.get('level', SEVERITY_UNKNOWN),
                            "message": f"Thuốc {drug_atc} và bệnh {icd_code}: {details.get('statement', '')}"
                        })
                    except json.JSONDecodeError:
                        print(f"Lỗi: Dữ liệu JSON cho ddxi:{drug_atc} với icd {icd_code} bị lỗi.")
        return alerts

    def _check_ddi(self, new_atcs, existing_atcs):
        alerts = []

        if len(new_atcs) >= 2:
            for drug_a, drug_b in combinations(new_atcs, 2):
                interaction_json = self.redis_conn.hget(f"ddi:{drug_a}", drug_b)
                if interaction_json:
                    try:
                        details = json.loads(interaction_json)
                        alerts.append({
                            "type": ALERT_TYPE_DDI,
                            "severity": details.get('severity', SEVERITY_UNKNOWN),
                            "message": f"[Trong đơn mới] Tương tác giữa {drug_a} và {drug_b}: {details.get('mechanism', '')}"
                        })
                    except json.JSONDecodeError:
                        print(f"Lỗi: Dữ liệu JSON cho ddi:{drug_a}|{drug_b} bị lỗi.")

        if existing_atcs:
            for new_drug, existing_drug in product(new_atcs, existing_atcs):
                if new_drug == existing_drug: continue
                interaction_json = self.redis_conn.hget(f"ddi:{new_drug}", existing_drug)
                if interaction_json:
                    try:
                        details = json.loads(interaction_json)
                        alerts.append({
                            "type": ALERT_TYPE_DDI,
                            "severity": details.get('severity', SEVERITY_UNKNOWN),
                            "message": f"[Mới-Cũ] Tương tác giữa thuốc mới kê ({new_drug}) và thuốc đang dùng ({existing_drug}): {details.get('mechanism', '')}"
                        })
                    except json.JSONDecodeError:
                        print(f"Lỗi: Dữ liệu JSON cho ddi:{new_drug}|{existing_drug} bị lỗi.")
        return alerts

    def _check_dose(self, prescription, patient_profile):
        alerts = []
        patient_age = patient_profile.get('age')

        for drug in prescription:
            atc_code = drug.get('atc_code')
            prescribed_dose = drug.get('dose')
            prescribed_unit = drug.get('unit')
            if not all([atc_code, prescribed_dose, prescribed_unit]):
                continue

            rules_json = self.redis_conn.get(f"dose:{atc_code}")
            if not rules_json:
                continue

            try:
                dose_rules = json.loads(rules_json)
            except json.JSONDecodeError:
                print(f"Lỗi: Dữ liệu JSON cho dose:{atc_code} bị lỗi.")
                continue

            relevant_rule = None
            if patient_age and patient_age >= 18:
                relevant_rule = next((rule for rule in dose_rules if rule.get('population') == 'adult'), None)

            if not relevant_rule and dose_rules:
                relevant_rule = dose_rules[0]

            if relevant_rule and prescribed_unit == relevant_rule.get('dose_unit'):
                dose_max = relevant_rule.get('dose_max')
                if dose_max is not None and prescribed_dose > dose_max:
                    alerts.append({
                        "type": ALERT_TYPE_DOSE,
                        "severity": SEVERITY_MAJOR,
                        "message": f"Liều dùng của {atc_code} ({prescribed_dose} {prescribed_unit}) vượt quá liều tối đa khuyến cáo ({dose_max} {relevant_rule.get('dose_unit')})."
                    })
        return alerts