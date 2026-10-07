# src/models/train.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

def main():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    with open("models/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)
        
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("Model training completed and saved to models/trained_model.pkl.")

if __name__ == "__main__":
    main()
