import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostForecaster:
    """
    A wrapper for the XGBoost Regressor for time-series forecasting.
    """
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=5):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            early_stopping_rounds=50,
            objective='reg:squarederror'
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame):
        """
        Separate features (X) and target (y).
        Excludes non-numeric or irrelevant columns like 'Date' and 'State'.
        """
        X = df.drop(columns=['Total', 'Date', 'State'], errors='ignore')
        y = df['Total'] if 'Total' in df.columns else None
        return X, y

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Fit the XGBoost model with early stopping.
        """
        print("Preparing features for XGBoost...")
        X_train, y_train = self._prepare_features(train_df)
        X_val, y_val = self._prepare_features(val_df)
        
        print(f"Fitting XGBoost with {X_train.shape[1]} features...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self.is_fitted = True
        print("Model fit complete.")

    def predict(self, df: pd.DataFrame):
        """
        Predict values for a given dataframe.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")
        
        X, _ = self._prepare_features(df)
        return self.model.predict(X)

    def get_feature_importance(self, feature_names):
        """
        Return the importance of each feature.
        """
        importances = self.model.feature_importances_
        return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

if __name__ == "__main__":
    # Test on dummy data
    data = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=100),
        'Total': np.random.rand(100) * 100,
        'State': ['CA'] * 100,
        'Lag_1': np.random.rand(100),
        'Month': np.random.randint(1, 13, 100)
    })
    
    train = data.iloc[:80]
    val = data.iloc[80:]
    
    forecaster = XGBoostForecaster()
    forecaster.fit(train, val)
    preds = forecaster.predict(val)
    print("\nSample Predictions:")
    print(preds[:5])
