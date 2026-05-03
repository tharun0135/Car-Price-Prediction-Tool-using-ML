import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_FILENAME = "car_price_pipeline.pkl"
DATA_FILENAME = "india_car_dataset_700.csv"

NUMERIC_FEATURES = [
    "year",
    "engine_cc",
    "horsepower",
    "mileage_kmpl",
    "engine_size",
    "range_km",
]

CATEGORICAL_FEATURES = [
    "fuel_type",
    "car_type",
    "engine_type",
    "fuel_system",
    "drive_wheel",
    "transmission",
    "cylinders",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(filepath=DATA_FILENAME):
    return pd.read_csv(filepath)


def build_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42, n_estimators=200)),
    ])


def predict_car_price(features: dict):
    model = joblib.load(MODEL_FILENAME)
    input_df = pd.DataFrame([features])[FEATURES]
    return float(model.predict(input_df)[0])


def main():
    df = load_data()
    X = df[FEATURES]
    y = df["price_lakh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model trained with {len(FEATURES)} mixed features.")
    print(f"Test R^2 score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    joblib.dump(model, MODEL_FILENAME)
    print(f"Saved trained model pipeline to {MODEL_FILENAME}")


if __name__ == "__main__":
    main()
