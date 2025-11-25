import pandas as pd
import psycopg2
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

POSTGRES_CONFIG = {
    "dbname": "cds_knowledge_base",
    "user": "cds_user",
    "password": "cds_password",
    "host": "localhost",
    "port": "5433"
}


class DataPreparator:
    def __init__(self, db_config):
        self.db_config = db_config

    def extract_data(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                df = pd.read_sql("SELECT * FROM event_logs", conn)
            return df
        except Exception as e:
            print(f"Lỗi khi trích xuất dữ liệu: {e}")
            return pd.DataFrame()

    def fit_and_transform(self, df, models_path):
        if df.empty:
            print("Bỏ qua vì không có dữ liệu.")
            return None, None

        os.makedirs(models_path, exist_ok=True)

        df['label'] = (df['doctor_action'] == 'CANCEL').astype(int)

        label_counts = df['label'].value_counts()
        print(f"   -> Phân phối nhãn (0/1): \n{label_counts}")

        mlb_atc = MultiLabelBinarizer()
        atc_features = mlb_atc.fit_transform(df['list_atc'])
        atc_df = pd.DataFrame(atc_features, columns=[f"atc_{c}" for c in mlb_atc.classes_])
        joblib.dump(mlb_atc, os.path.join(models_path, 'mlb_atc.joblib'))  # <-- LƯU LẠI

        mlb_icd = MultiLabelBinarizer()
        icd_features = mlb_icd.fit_transform(df['icd_list'])
        icd_df = pd.DataFrame(icd_features, columns=[f"icd_{c}" for c in mlb_icd.classes_])
        joblib.dump(mlb_icd, os.path.join(models_path, 'mlb_icd.joblib'))  # <-- LƯU LẠI

        categorical_features = ['alert_type', 'severity', 'patient_gender']
        numerical_features = ['patient_age']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'
        )
        processed_features = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, os.path.join(models_path, 'column_transformer.joblib'))

        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(cat_feature_names)
        processed_df = pd.DataFrame(processed_features, columns=all_feature_names)

        final_df = pd.concat([processed_df, atc_df, icd_df], axis=1)

        feature_names = final_df.columns.tolist()
        joblib.dump(feature_names, os.path.join(models_path, 'feature_names.joblib'))

        print(f"Biến đổi hoàn tất. Tổng số features: {final_df.shape[1]}")
        return final_df, df['label']

    def load_data(self, X, y, parquet_path, csv_path):
        if X is None or y is None: return
        try:
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            data_to_save = X.copy()
            data_to_save['label'] = y
            data_to_save.to_parquet(parquet_path, index=False, engine='pyarrow')
            print(f"Đã lưu file Parquet thành công tại: {parquet_path}")
            data_to_save.to_csv(csv_path, index=False)
            print(f"Đã lưu file CSV thành công tại: {csv_path}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu: {e}")

    def run(self, models_path, parquet_path, csv_path):
        raw_data = self.extract_data()
        X, y = self.fit_and_transform(raw_data, models_path)
        self.load_data(X, y, parquet_path, csv_path)


if __name__ == '__main__':
    preparator = DataPreparator(POSTGRES_CONFIG)
    preparator.run(
        models_path='../models',
        parquet_path='../data/processed/training_data.parquet',
        csv_path='../data/processed/training_data.csv'
    )