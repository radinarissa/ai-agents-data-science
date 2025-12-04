import os
import pandas as pd   # ✅ import pandas at the top

class DataProcessingAgent:
    def __init__(self, df):
        self.df = df
        self.cleaned = None
        self.features = None
        self.validation = None

    def run_pipeline(self, contamination=0.05, impute_strategy="mean", feature_depth=1):
        print("🔹 Starting Data Processing Pipeline...")

        from agents.data_processing import DataCleaner, FeatureEngineer, DataValidator
       
        cleaner = DataCleaner(self.df)
        self.cleaned = cleaner.detect_outliers(
            contamination=contamination,
            impute_strategy=impute_strategy
        )

        fe = FeatureEngineer(self.cleaned)
        self.features = fe.generate_features(max_depth=feature_depth)
        self.features = self.features.rename(columns={
            "data_quuality_score": "data_quality_score"
        })
        
        validator = DataValidator(self.features)
        self.validation = validator.validate()

        # Remove constant and low-variance columns
        const_cols = self.validation[self.validation["is_constant"] == True].index.tolist()
        if const_cols:
            print(f"🗑 Removing constant columns: {const_cols}")
            self.features = self.features.drop(columns=const_cols)

        low_var_cols = self.validation[self.validation.get("low_variance") == True].index.tolist()
        if low_var_cols:
            print(f"⚠️ Removing low variance columns: {low_var_cols}")
            self.features = self.features.drop(columns=low_var_cols)

        # Drop raw timestamp and exploded timestamp features
        self.features = self.features.drop(
            columns=[c for c in self.features.columns if "timestamp" in c.lower() or "(" in c],
            errors="ignore"
        )

        # Encode categorical features
        categorical_cols = self.features.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            print(f"🔤 Encoding categorical columns: {list(categorical_cols)}")
            self.features = pd.get_dummies(self.features, columns=categorical_cols, drop_first=True)

        # Detect and drop high-cardinality dummy features
        high_card_cols = [col for col in self.features.columns if self.features[col].nunique() > 50]
        if high_card_cols:
            print(f"🗑 Dropping high-cardinality features: {len(high_card_cols)} columns")
            self.features = self.features.drop(columns=high_card_cols)

        # Final check: ensure all columns are numeric
        non_numeric = self.features.select_dtypes(exclude=["number"]).columns
        if len(non_numeric) > 0:
            print(f"⚠️ Warning: Non-numeric columns still present: {list(non_numeric)}")
            self.features[non_numeric] = self.features[non_numeric].astype(str)
            self.features = pd.get_dummies(self.features, columns=non_numeric, drop_first=True)

        # Preserve target column
        if "success_rate" in self.cleaned.columns and "success_rate" not in self.features.columns:
            self.features["success_rate"] = self.cleaned["success_rate"].values

        print("✅ Pipeline completed successfully.")

        os.makedirs("experiments", exist_ok=True)

        self.features.to_csv("data/processed/processed_dataset.csv", index=False)
        self.validation.to_csv("experiments/validation_report.csv")

        with open("experiments/cleaning_log.txt", "w") as f:
            f.write("Data Processing Pipeline Log\n")
            f.write("----------------------------------\n")
            f.write(f"Outlier contamination: {contamination}\n")
            f.write(f"Imputation strategy: {impute_strategy}\n")
            f.write(f"Feature depth: {feature_depth}\n")
            f.write("\nRemoved constant columns:\n")
            f.write(str(const_cols) + "\n")
            f.write("\nRemoved low variance columns:\n")
            f.write(str(low_var_cols) + "\n")
            f.write("\nEncoded categorical columns:\n")
            f.write(str(list(categorical_cols)) + "\n")
            f.write("\nDropped high-cardinality columns:\n")
            f.write(str(high_card_cols) + "\n")
            f.write("\nFinal feature set shape:\n")
            f.write(str(self.features.shape) + "\n")

        print("💾 Saved processed dataset to /data/processed/; validation report and logs to /experiments/")

        return self.features, self.validation
