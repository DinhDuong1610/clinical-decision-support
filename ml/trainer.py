import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    def __init__(self, data_path, models_path, reports_path):
        self.data_path = data_path
        self.models_path = models_path
        self.reports_path = reports_path
        self.model = None

    def load_data(self):
        print("1. [Load] Bắt đầu tải dữ liệu đã xử lý...")
        try:
            df = pd.read_parquet(self.data_path)
            X = df.drop('label', axis=1)
            y = df['label']
            print(f"Tải thành công. Dữ liệu có {X.shape[0]} mẫu và {X.shape[1]} features.")
            return X, y
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            return None, None

    def train(self, X, y):
        if X is None or y is None:
            print("Bỏ qua huấn luyện vì không có dữ liệu.")
            return

        print("2. [Train] Bắt đầu phân chia dữ liệu...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu.")
        print(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu.")

        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0]
        }

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1

        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            verbose=2,
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Các tham số tốt nhất tìm được: {grid_search.best_params_}")

        self.model = grid_search.best_estimator_

        print("\n4. [Evaluate] Bắt đầu đánh giá mô hình TỐI ƯU trên tập kiểm tra...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n--- Báo cáo ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("------------------------")

        self._visualize_results(X_test, y_test, y_pred)

    def _visualize_results(self, X_test, y_test, y_pred):
        figures_path = os.path.join(self.reports_path, 'figures')
        os.makedirs(figures_path, exist_ok=True)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Nhiễu (0)', 'Hữu ích (1)'],
                    yticklabels=['Nhiễu (0)', 'Hữu ích (1)'])
        plt.xlabel('Giá trị Dự đoán')
        plt.ylabel('Giá trị Thực tế')
        plt.title('Ma trận Nhầm lẫn')
        confusion_matrix_path = os.path.join(figures_path, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Đã lưu Ma trận Nhầm lẫn tại: {confusion_matrix_path}")

        feature_importances = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances, hue='feature', palette='viridis', legend=False)
        plt.title('15 Features Quan trọng nhất')
        plt.xlabel('Mức độ Quan trọng')
        plt.ylabel('Feature')
        plt.tight_layout()
        feature_importance_path = os.path.join(figures_path, 'feature_importance.png')
        plt.savefig(feature_importance_path)
        plt.close()
        print(f"Đã lưu biểu đồ Feature Importance tại: {feature_importance_path}")

    def save_model(self):
        if self.model is None:
            print("\n6. [Save] Bỏ qua vì mô hình chưa được huấn luyện.")
            return

        print("\n6. [Save] Bắt đầu lưu mô hình đã huấn luyện...")
        try:
            os.makedirs(self.models_path, exist_ok=True)
            model_path = os.path.join(self.models_path, 'xgb_model_tuned.joblib')
            joblib.dump(self.model, model_path)
            print(f"Đã lưu mô hình thành công tại: {model_path}")
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")

    def run(self):
        X, y = self.load_data()
        self.train(X, y)
        self.save_model()

if __name__ == '__main__':
    trainer = ModelTrainer(
        data_path='../data/processed/training_data.parquet',
        models_path='../models',
        reports_path='../reports'
    )
    trainer.run()