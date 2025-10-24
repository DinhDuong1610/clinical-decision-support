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
        self.champion_model_path = os.path.join(self.models_path, 'xgb_model_tuned.joblib')
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

    def train_new_challenger(self, X_train, y_train):
        print("\n3. [Train Challenger] Bắt đầu huấn luyện mô hình mới...")

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
            scoring='roc_auc', cv=5,
            verbose=0,
            n_jobs=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Các tham số tốt nhất tìm được: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _evaluate_model(self, model, X_test, y_test, model_name="Model"):
        print(f"\n--- Đánh giá cho {model_name} ---")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred, zero_division=0))
        score = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC Score: {score:.4f}")
        print("------------------------")
        return score, y_pred

    def _champion_challenger_evaluation(self, challenger_model, X_test, y_test):
        print("\n4. [Champion vs. Challenger] Bắt đầu so sánh hiệu năng...")

        challenger_score, y_pred = self._evaluate_model(challenger_model, X_test, y_test, "Challenger (Mô hình mới)")

        if os.path.exists(self.champion_model_path):
            print(f"\nTải mô hình Champion hiện tại từ: {self.champion_model_path}")
            champion_model = joblib.load(self.champion_model_path)
            champion_score, _ = self._evaluate_model(champion_model, X_test, y_test, "Champion (Mô hình cũ)")
        else:
            print("\nKhông tìm thấy mô hình Champion. Mô hình mới sẽ tự động trở thành Champion.")
            champion_score = -1.0

        if challenger_score > champion_score:
            print(f"\nKẾT QUẢ: Challenger THẮNG! (Score: {challenger_score:.4f} > {champion_score:.4f})")
            print("Triển khai mô hình mới...")
            self.model = challenger_model
            self._visualize_results(X_test, y_test, y_pred)
            self.save_model()
        else:
            print(f"\nKẾT QUẢ: Champion BẢO VỆ THÀNH CÔNG! (Score: {challenger_score:.4f} <= {champion_score:.4f})")
            print("Giữ lại mô hình cũ.")

    def _visualize_results(self, X_test, y_test, y_pred):
        print("\n5. [Visualize] Bắt đầu tạo các biểu đồ trực quan hóa...")
        figures_path = os.path.join(self.reports_path, 'figures')
        os.makedirs(figures_path, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nhiễu (0)', 'Hữu ích (1)'],
                    yticklabels=['Nhiễu (0)', 'Hữu ích (1)'])
        plt.xlabel('Giá trị Dự đoán');
        plt.ylabel('Giá trị Thực tế');
        plt.title('Ma trận Nhầm lẫn')
        plt.savefig(os.path.join(figures_path, 'confusion_matrix.png'));
        plt.close()
        print(f"Đã lưu Ma trận Nhầm lẫn.")
        feature_importances = pd.DataFrame(
            {'feature': X_test.columns, 'importance': self.model.feature_importances_}).sort_values('importance',
                                                                                                    ascending=False).head(
            15)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances, hue='feature', palette='viridis',
                    legend=False)
        plt.title('15 Features Quan trọng nhất');
        plt.xlabel('Mức độ Quan trọng');
        plt.ylabel('Feature');
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, 'feature_importance.png'));
        plt.close()
        print(f"Đã lưu biểu đồ Feature Importance.")

    def save_model(self):
        if self.model is None: return
        print("\n6. [Save] Bắt đầu lưu mô hình mới...")
        try:
            joblib.dump(self.model, self.champion_model_path)
            print(f"Đã lưu mô hình thành công tại: {self.champion_model_path}")
        except Exception as e:
            print(f"Lỗi khi lưu mô hình: {e}")

    def run(self):
        X, y = self.load_data()
        if X is None or y is None: return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"2. [Data Split] Đã phân chia dữ liệu.")
        print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu.")
        print(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu.")

        challenger_model = self.train_new_challenger(X_train, y_train)

        self._champion_challenger_evaluation(challenger_model, X_test, y_test)

if __name__ == '__main__':
    trainer = ModelTrainer(
        data_path='../data/processed/training_data.parquet',
        models_path='../models',
        reports_path='../reports'
    )
    trainer.run()