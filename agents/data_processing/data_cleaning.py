import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def handle_missing(self, strategy="mean"):
        num_cols = self.df.select_dtypes(include=np.number).columns
        imputer = SimpleImputer(strategy=strategy)
        self.df[num_cols] = imputer.fit_transform(self.df[num_cols])
        return self.df

    def detect_outliers(self, contamination=0.05, impute_strategy="mean"):
        num_cols = self.df.select_dtypes(include=np.number).columns
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(self.df[num_cols])
        outliers = np.where(preds == -1)[0]
        print(f"⚠️ Detected {len(outliers)} outliers out of {len(self.df)} rows.")
        self.df.loc[outliers, num_cols] = np.nan
        self.df = self.handle_missing(strategy=impute_strategy)
        return self.df
