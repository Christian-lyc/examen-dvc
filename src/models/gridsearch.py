# src/models/gridsearch.py
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def main():
    os.makedirs("models", exist_ok=True)
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(grid_search.best_params_, f)
        
    print(f"Best parameters saved: {grid_search.best_params_}")

if __name__ == "__main__":
    main()
