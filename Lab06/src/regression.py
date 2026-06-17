from pathlib import Path
import os
import tempfile

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "lab06_matplotlib_cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


PRICE_LOWER_QUANTILE = 0.01
PRICE_UPPER_QUANTILE = 0.995


def make_one_hot_encoder():
    """Tạo OneHotEncoder tương thích với nhiều phiên bản scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    df = pd.read_csv(path)
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_rows = len(df)

    df["yr_renovated"] = df["yr_renovated"].fillna(0)

    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    numeric_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[df["price"] > 0]
    df = df[df["bedrooms"] >= 0]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month

    df["house_age"] = df["sale_year"] - df["yr_built"]
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

    df = df.dropna()

    lower_price = df["price"].quantile(PRICE_LOWER_QUANTILE)
    upper_price = df["price"].quantile(PRICE_UPPER_QUANTILE)
    before_outlier_filter = len(df)
    df = df[(df["price"] >= lower_price) & (df["price"] <= upper_price)]

    removed_invalid_rows = original_rows - before_outlier_filter
    removed_outlier_rows = before_outlier_filter - len(df)

    print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")
    print(f"Số dòng lỗi/thiếu/bất hợp lý đã loại: {removed_invalid_rows}")
    print(
        "Số dòng outlier về giá đã loại "
        f"({PRICE_LOWER_QUANTILE:.1%}-{PRICE_UPPER_QUANTILE:.1%}): "
        f"{removed_outlier_rows}"
    )
    print(f"Khoảng giá giữ lại: {lower_price:,.0f} - {upper_price:,.0f}")
    return df


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["id", "date", "price"])
    y = df["price"]

    X["zipcode"] = X["zipcode"].astype(str)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
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
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Decision Tree No Limit": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
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


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return {
        "Model": model_name,
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train RMSE": np.sqrt(train_mse),
        "Test RMSE": np.sqrt(test_mse),
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
    }


def train_and_evaluate_models(X, y, numeric_features, categorical_features):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = get_model_estimators()

    trained_models = {}
    results = []

    for model_name, estimator in models.items():
        print(f"Đang huấn luyện: {model_name}")
        model = make_model_pipeline(estimator, numeric_features, categorical_features)

        model.fit(X_train, y_train)
        trained_models[model_name] = model

        result = evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values(by="Test RMSE")
    return results_df, trained_models, X_train, y_train, X_test, y_test


def save_comparison_chart(results_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Test RMSE")
    plt.title("So sánh Test RMSE giữa các mô hình")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Test R2")
    plt.title("So sánh Test R2 giữa các mô hình")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_r2.png", dpi=150)
    plt.close()


def save_random_forest_feature_importance(model: Pipeline, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    feature_importance_df.to_csv(
        output_dir / "random_forest_feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance_df.head(15),
        x="Importance",
        y="Feature",
    )
    plt.title("Top 15 đặc trưng quan trọng nhất - Random Forest")
    plt.tight_layout()
    plt.savefig(output_dir / "random_forest_feature_importance.png", dpi=150)
    plt.close()


def save_sample_predictions(model: Pipeline, X_test, y_test, output_dir: Path):
    sample = X_test.head(10)
    actual_price = y_test.head(10)
    predicted_price = model.predict(sample)

    comparison_df = pd.DataFrame(
        {
            "Actual Price": actual_price.values,
            "Predicted Price": predicted_price,
            "Absolute Error": np.abs(actual_price.values - predicted_price),
        }
    )

    comparison_df.to_csv(output_dir / "sample_predictions.csv", index=False)


def make_filename_slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def save_learning_curve_charts(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)

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
                    "Train RMSE": result["Train RMSE"],
                    "Test RMSE": result["Test RMSE"],
                    "Train R2": result["Train R2"],
                    "Test R2": result["Test R2"],
                }
            )

    learning_curve_df = pd.DataFrame(rows)
    learning_curve_df.to_csv(output_dir / "learning_curves_all_models.csv", index=False)

    for model_name in learning_curve_df["Model"].unique():
        model_df = learning_curve_df[learning_curve_df["Model"] == model_name]
        slug = make_filename_slug(model_name)

        plt.figure(figsize=(10, 5))
        plt.plot(
            model_df["Train Size"],
            model_df["Train RMSE"],
            marker="o",
            label="Train RMSE",
        )
        plt.plot(
            model_df["Train Size"],
            model_df["Test RMSE"],
            marker="o",
            label="Test RMSE",
        )
        plt.xlabel("Số lượng mẫu train")
        plt.ylabel("RMSE")
        plt.title(f"Learning Curve RMSE - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"learning_curve_{slug}_rmse.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(
            model_df["Train Size"],
            model_df["Train R2"],
            marker="o",
            label="Train R2",
        )
        plt.plot(
            model_df["Train Size"],
            model_df["Test R2"],
            marker="o",
            label="Test R2",
        )
        plt.xlabel("Số lượng mẫu train")
        plt.ylabel("R2 Score")
        plt.title(f"Learning Curve R2 - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"learning_curve_{slug}_r2.png", dpi=150)
        plt.close()


def save_overfit_underfit_charts(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_features,
    categorical_features,
    output_dir: Path,
):
    output_dir.mkdir(exist_ok=True)

    max_depth_values = [1, 2, 3, 5, 7, 10, 15, 20, None]
    rows = []

    for max_depth in max_depth_values:
        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numeric_features, categorical_features),
                ),
                (
                    "model",
                    DecisionTreeRegressor(max_depth=max_depth, random_state=42),
                ),
            ]
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
                "Train RMSE": result["Train RMSE"],
                "Test RMSE": result["Test RMSE"],
                "Train R2": result["Train R2"],
                "Test R2": result["Test R2"],
            }
        )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_dir / "overfit_underfit_decision_tree.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(
        curve_df["complexity_index"],
        curve_df["Train RMSE"],
        marker="o",
        label="Train RMSE",
    )
    plt.plot(
        curve_df["complexity_index"],
        curve_df["Test RMSE"],
        marker="o",
        label="Test RMSE",
    )
    plt.xticks(curve_df["complexity_index"], curve_df["max_depth"])
    plt.xlabel("max_depth của Decision Tree")
    plt.ylabel("RMSE")
    plt.title("Nhận diện underfitting/overfitting bằng RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_underfit_decision_tree_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        curve_df["complexity_index"],
        curve_df["Train R2"],
        marker="o",
        label="Train R2",
    )
    plt.plot(
        curve_df["complexity_index"],
        curve_df["Test R2"],
        marker="o",
        label="Test R2",
    )
    plt.xticks(curve_df["complexity_index"], curve_df["max_depth"])
    plt.xlabel("max_depth của Decision Tree")
    plt.ylabel("R2 Score")
    plt.title("Nhận diện underfitting/overfitting bằng R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overfit_underfit_decision_tree_r2.png", dpi=150)
    plt.close()


def run_regression_experiment(data_path: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    df = load_data(data_path)
    df = clean_data(df)
    X, y, numeric_features, categorical_features = prepare_features(df)

    results_df, trained_models, X_train, y_train, X_test, y_test = train_and_evaluate_models(
        X,
        y,
        numeric_features,
        categorical_features,
    )

    print("\nBảng so sánh mô hình:")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_dir / "model_results.csv", index=False)
    save_comparison_chart(results_df, output_dir)

    best_forest = trained_models["Random Forest"]
    save_random_forest_feature_importance(best_forest, output_dir)
    save_sample_predictions(best_forest, X_test, y_test, output_dir)
    save_learning_curve_charts(
        X_train,
        y_train,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        output_dir,
    )
    save_overfit_underfit_charts(
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
