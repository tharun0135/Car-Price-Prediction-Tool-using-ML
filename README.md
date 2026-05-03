# Car Price Prediction Dashboard using ML

A modern web application for predicting car prices using machine learning and exploring market trends through interactive dashboards.

## Features

- **Price Prediction**: Input car specifications (like engine CC, horsepower, and transmission) to get accurate price estimates in Lakhs.
- **Personalized Suggestions**: Set budget parameters to get car recommendations within your specified range, sorted by closest price match.
- **Interactive Dashboard**: Visualize how different features affect car prices with dynamic line and bar charts.
- **Advanced Filtering**: Search for cars in the database using multiple filters including fuel type, transmission, and mileage.
- **Modern UI**: Clean, responsive, and futuristic "diagnostic HUD" design with gradient backgrounds and smooth animations.

## Tech Stack & Tools

- **Backend Framework**: FastAPI (Python) for building the API and serving the web interface.
- **Data Manipulation**: Pandas for loading and processing CSV datasets.
- **Machine Learning**: Scikit-learn for preprocessing pipelines and training the model.
- **Model Serialization**: Joblib for saving and loading the trained model pipeline (`car_price_pipeline.pkl`).
- **Frontend**: HTML5, CSS3, and vanilla JavaScript, served directly via a FastAPI `HTMLResponse`.
- **Data Visualization**: Chart.js for rendering interactive price trend and fuel-type charts.
- **Server**: Uvicorn for running the FastAPI application.
- **Data Validation**: Pydantic.

## Machine Learning Pipeline

The predictive engine uses a Scikit-learn `ColumnTransformer` to handle different types of data simultaneously before feeding them into a regression model.

### 1. Data Preprocessing
- **Numeric Features** (`year`, `engine_cc`, `horsepower`, `mileage_kmpl`, `engine_size`, `range_km`):
  - **Imputation**: Missing values are filled using the median value (`SimpleImputer(strategy="median")`).
  - **Scaling**: Standardized to have a mean of 0 and a variance of 1 (`StandardScaler`).
- **Categorical Features** (`fuel_type`, `car_type`, `engine_type`, `fuel_system`, `drive_wheel`, `transmission`, `cylinders`):
  - **Imputation**: Missing values are filled using the mode/most frequent value (`SimpleImputer(strategy="most_frequent")`).
  - **Encoding**: Converted into binary vectors (`OneHotEncoder(handle_unknown="ignore")`).

### 2. Predictive Model
- **Algorithm**: `RandomForestRegressor` for continuous price variable prediction.
- **Hyperparameters**: Configured to build 200 decision trees (`n_estimators=200`) with a fixed `random_state=42` for reproducibility.

### 3. Evaluation Metrics
The model's performance on the testing set (20% of the dataset) is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and the R-squared score.

## Dataset Specification

This application uses the `cleaned_cars_dataset.csv` file for training and data retrieval. Please ensure this file is placed in the root directory before training the model or running the server.

## Installation & Setup

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn fastapi uvicorn joblib pydantic

Train the machine learning pipeline (This generates the car_price_pipeline.pkl file):

Bash
python train_model.py

Run the application:

Bash
uvicorn app:app --reload

Open http://127.0.0.1:8000 in your web browser.

Cloud Deployment
Ensure you use the following start command for cloud environments:

Bash
uvicorn app:app --host 0.0.0.0 --port $PORT
