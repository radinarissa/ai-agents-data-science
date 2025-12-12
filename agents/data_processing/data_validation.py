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

        numeric = self.df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary.loc[numeric.columns, "variance"] = numeric.var()

            summary["low_variance"] = summary["variance"] < 0.01

            summary.loc[numeric.columns, "num_negatives"] = (numeric < 0).sum()

        if "timestamp" in self.df.columns:
            summary.loc["timestamp", "is_datetime"] = True
        numeric = self.df.select_dtypes(include=["number"])
        if not numeric.empty:
            summary.loc[numeric.columns, "variance"] = numeric.var()

            summary["low_variance"] = summary["variance"] < 0.01

            summary.loc[numeric.columns, "num_negatives"] = (numeric < 0).sum()

        if "timestamp" in self.df.columns:
            summary.loc["timestamp", "is_datetime"] = True

        print("✅ Data validation complete.")
        return summary
