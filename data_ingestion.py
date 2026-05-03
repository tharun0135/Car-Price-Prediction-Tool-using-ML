import pandas as pd

DATA_FILENAME = "india_car_dataset_700.csv"


def load_data(filepath=DATA_FILENAME):
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it is in the root directory.")
        return None


if __name__ == "__main__":
    df = load_data()
