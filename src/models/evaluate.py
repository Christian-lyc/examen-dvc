# src/models/evaluate.py
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

def main():
    os.makedirs("metrics", exist_ok=True)
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    
    pd.DataFrame({"predictions": y_pred}).to_csv("data/predictions.csv", index=False)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    scores = {
        "mse": mse,
        "r2": r2
    }
    
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
        
    print(f"Evaluation completed. Scores: {scores}")

if __name__ == "__main__":
    main()
