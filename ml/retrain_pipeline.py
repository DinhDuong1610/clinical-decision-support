import os
from datetime import datetime

from data_preparator import DataPreparator, POSTGRES_CONFIG
from trainer import ModelTrainer

def run_retraining_pipeline():
    print("=========================================================")
    print(f"BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN LẠI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================================")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'processed')
    models_path = os.path.join(base_dir, 'models')
    reports_path = os.path.join(base_dir, 'reports')

    print("\n[PIPELINE] Giai đoạn 1: Chuẩn bị Dữ liệu")
    preparator = DataPreparator(POSTGRES_CONFIG)
    preparator.run(
        models_path=models_path,
        parquet_path=os.path.join(data_path, 'training_data.parquet'),
        csv_path=os.path.join(data_path, 'training_data.csv')
    )

    print("\n[PIPELINE] Giai đoạn 2: Huấn luyện Mô hình")
    trainer = ModelTrainer(
        data_path=os.path.join(data_path, 'training_data.parquet'),
        models_path=models_path,
        reports_path=reports_path
    )
    trainer.run()

    print("\n=========================================================")
    print(f"QUY TRÌNH HUẤN LUYỆN LẠI HOÀN TẤT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================================")


if __name__ == '__main__':
    run_retraining_pipeline()