import pandas as pd

class DataValidator:
    def __init__(self, df):
        self.df = df

    def validate(self):
        summary = pd.DataFrame({
            "dtype": self.df.dtypes.astype(str),
            "missing_%": self.df.isnull().mean() * 100,
            "unique_values": self.df.nunique(),
        })

        summary["is_constant"] = summary["unique_values"] == 1

<<<<<<< HEAD
        numeric = self.df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary.loc[numeric.columns, "variance"] = numeric.var()

            summary["low_variance"] = summary["variance"] < 0.01

            summary.loc[numeric.columns, "num_negatives"] = (numeric < 0).sum()

        if "timestamp" in self.df.columns:
            summary.loc["timestamp", "is_datetime"] = True
=======
       
        numeric = self.df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary.loc[numeric.columns, "variance"] = numeric.var()
            summary["low_variance"] = summary["variance"] < 0.01

       
        if "Age" in self.df.columns:
            summary.loc["Age", "impossible_values"] = (self.df["Age"] < 0).sum()

        if "Fare" in self.df.columns:
            summary.loc["Fare", "impossible_values"] = (self.df["Fare"] < 0).sum()
>>>>>>> cd9890f0e46783a92ea6995fb76c91f747bd3ae1

        print("✅ Data validation complete.")
        return summary
