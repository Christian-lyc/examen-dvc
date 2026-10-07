# src/data/split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/raw.csv")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    X = numeric_df.drop(columns=["silica_concentrate"])
    y = numeric_df["silica_concentrate"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("Data split completed successfully.")

if __name__ == "__main__":
    main()
