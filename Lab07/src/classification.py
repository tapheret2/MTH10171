from pathlib import Path
import os
import tempfile

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "lab07_matplotlib_cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_CANDIDATES = ["loan_status", "Loan_Status", "status", "Status"]
POSITIVE_CLASS = 1


def make_one_hot_encoder():
    """Tạo OneHotEncoder tương thích với nhiều phiên bản scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu. Hãy đặt file Loan Approval Prediction Dataset "
            f"tại {path}"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")
    return df


def find_target_column(df: pd.DataFrame) -> str:
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Không tìm thấy cột mục tiêu. Hãy kiểm tra lại tên cột loan_status."
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()

    df = df.drop_duplicates()
    df = df.dropna()

    print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")
    return df


def encode_target(y: pd.Series) -> pd.Series:
    y_clean = y.astype(str).str.strip().str.lower()

    mapping = {
        "approved": 1,
        "approve": 1,
        "yes": 1,
        "y": 1,
        "1": 1,
        "rejected": 0,
        "reject": 0,
        "no": 0,
        "n": 0,
        "0": 0,
    }

    y_encoded = y_clean.map(mapping)

    if y_encoded.isnull().any():
        unknown_values = sorted(y[y_encoded.isnull()].unique())
        raise ValueError(f"Giá trị nhãn chưa hỗ trợ: {unknown_values}")

    return y_encoded.astype(int)


def prepare_features(df: pd.DataFrame):
    target_column = find_target_column(df)

    X = df.drop(columns=[target_column])
    y = encode_target(df[target_column])

    id_columns = [column for column in X.columns if column.lower() in ["loan_id", "id"]]
    X = X.drop(columns=id_columns)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ]
    )


def get_model_estimators():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Decision Tree No Limit": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
    }


def make_model_pipeline(estimator, numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", clone(estimator)),
        ]
    )


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, POSITIVE_CLASS]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X)


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_score = predict_scores(model, X_test)

    return {
        "Model": model_name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Train Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "Test Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "Train F1": f1_score(y_train, y_train_pred, zero_division=0),
        "Test F1": f1_score(y_test, y_test_pred, zero_division=0),
        "Test ROC AUC": roc_auc_score(y_test, y_test_score),
    }


def train_and_evaluate_models(X, y, numeric_features, categorical_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trained_models = {}
    results = []

    for model_name, estimator in get_model_estimators().items():
        print(f"Đang huấn luyện: {model_name}")
        model = make_model_pipeline(estimator, numeric_features, categorical_features)
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        result = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="Test F1", ascending=False)
    return results_df, trained_models, X_train, y_train, X_test, y_test


def make_filename_slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def save_confusion_matrices(trained_models, X_test, y_test, output_dir: Path):
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["Rejected", "Approved"],
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"confusion_matrix_{make_filename_slug(model_name)}.png",
            dpi=150,
        )
        plt.close()


def save_classification_reports(trained_models, X_test, y_test, output_dir: Path):
    rows = []

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Rejected", "Approved"],
            output_dict=True,
            zero_division=0,
        )

        for label in ["Rejected", "Approved", "macro avg", "weighted avg"]:
            rows.append(
                {
                    "Model": model_name,
                    "Label": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1": report[label]["f1-score"],
                    "Support": report[label]["support"],
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "classification_reports.csv", index=False)


def save_roc_curves(trained_models, X_test, y_test, output_dir: Path):
    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models.items():
        y_score = predict_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()


def save_model_comparison_charts(results_df: pd.DataFrame, output_dir: Path):
    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC AUC"]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results_df, x="Model", y=metric)
        plt.title(f"So sánh {metric}")
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{make_filename_slug(metric)}.png")
        plt.close()


def save_learning_curve_charts(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    output_dir: Path,
):
    train_size_ratios = np.linspace(0.1, 1.0, 10)
    rows = []

    for model_name, estimator in get_model_estimators().items():
        print(f"Đang tạo learning curve: {model_name}")

        for ratio in train_size_ratios:
            sample_size = max(10, int(len(X_train) * ratio))
            X_subset = X_train.iloc[:sample_size]
            y_subset = y_train.iloc[:sample_size]

            model = make_model_pipeline(
                estimator,
                numeric_features,
                categorical_features,
            )
            model.fit(X_subset, y_subset)

            result = evaluate_model(
                model_name,
                model,
                X_subset,
                y_subset,
                X_test,
                y_test,
            )

            rows.append(
                {
                    "Model": model_name,
                    "Train Size Ratio": ratio,
                    "Train Size": sample_size,
                    "Train Accuracy": result["Train Accuracy"],
                    "Test Accuracy": result["Test Accuracy"],
                    "Train F1": result["Train F1"],
                    "Test F1": result["Test F1"],
                    "Test ROC AUC": result["Test ROC AUC"],
                }
            )

    learning_curve_df = pd.DataFrame(rows)
    learning_curve_df.to_csv(output_dir / "learning_curves_all_models.csv", index=False)

    for model_name in learning_curve_df["Model"].unique():
        model_df = learning_curve_df[learning_curve_df["Model"] == model_name]
        slug = make_filename_slug(model_name)

        for metric in ["F1", "Accuracy"]:
            plt.figure(figsize=(10, 5))
            plt.plot(
                model_df["Train Size"],
                model_df[f"Train {metric}"],
                marker="o",
                label=f"Train {metric}",
            )
            plt.plot(
                model_df["Train Size"],
                model_df[f"Test {metric}"],
                marker="o",
                label=f"Test {metric}",
            )
            plt.xlabel("Số lượng mẫu train")
            plt.ylabel(metric)
            plt.ylim(0, 1)
            plt.title(f"Learning Curve {metric} - {model_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                output_dir / f"learning_curve_{slug}_{metric.lower()}.png",
                dpi=150,
            )
            plt.close()


def save_decision_tree_validation_curve(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    output_dir: Path,
):
    max_depth_values = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
    rows = []

    for max_depth in max_depth_values:
        model = make_model_pipeline(
            DecisionTreeClassifier(max_depth=max_depth, random_state=42),
            numeric_features,
            categorical_features,
        )
        model.fit(X_train, y_train)
        result = evaluate_model(
            f"Decision Tree max_depth={max_depth}",
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        rows.append(
            {
                "max_depth": "None" if max_depth is None else max_depth,
                "complexity_index": len(rows) + 1,
                "Train Accuracy": result["Train Accuracy"],
                "Test Accuracy": result["Test Accuracy"],
                "Train F1": result["Train F1"],
                "Test F1": result["Test F1"],
                "Test ROC AUC": result["Test ROC AUC"],
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_dir / "validation_curve_decision_tree.csv", index=False)

    for metric in ["F1", "Accuracy"]:
        plt.figure(figsize=(10, 5))
        plt.plot(
            curve_df["complexity_index"],
            curve_df[f"Train {metric}"],
            marker="o",
            label=f"Train {metric}",
        )
        plt.plot(
            curve_df["complexity_index"],
            curve_df[f"Test {metric}"],
            marker="o",
            label=f"Test {metric}",
        )
        plt.xticks(curve_df["complexity_index"], curve_df["max_depth"])
        plt.xlabel("max_depth của Decision Tree")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.title(f"Validation Curve Decision Tree - {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"validation_curve_decision_tree_{metric.lower()}.png",
            dpi=150,
        )
        plt.close()


def run_classification_experiment(data_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)
    X, y, numeric_features, categorical_features = prepare_features(df)

    print("Phân bố nhãn:")
    print(y.value_counts().rename(index={0: "Rejected", 1: "Approved"}))

    results_df, trained_models, X_train, y_train, X_test, y_test = (
        train_and_evaluate_models(X, y, numeric_features, categorical_features)
    )

    print("\nBảng so sánh mô hình:")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_dir / "model_results.csv", index=False)
    save_model_comparison_charts(results_df, output_dir)
    save_confusion_matrices(trained_models, X_test, y_test, output_dir)
    save_classification_reports(trained_models, X_test, y_test, output_dir)
    save_roc_curves(trained_models, X_test, y_test, output_dir)
    save_learning_curve_charts(
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        output_dir,
    )
    save_decision_tree_validation_curve(
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        output_dir,
    )

    print(f"\nĐã lưu kết quả vào thư mục {output_dir}/")
    return results_df

