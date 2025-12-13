import os
import pandas as pd
from agents.data_processing import DataCleaner, FeatureEngineer, DataValidator

class DataProcessingAgent:
    def __init__(self, df):
        self.df = df
        self.cleaned = None
        self.features = None
        self.validation = None

    def run_pipeline(self, contamination=0.05, impute_strategy="mean", feature_depth=1):
        print("🔹 Starting Data Processing Pipeline...")

        # -------------------------------
        # Experiment 7: Data Cleaning – Outlier Detection
        # -------------------------------

        cleaner = DataCleaner(self.df)
        self.cleaned = cleaner.detect_outliers(
            contamination=contamination,
            impute_strategy=impute_strategy
        )

        # ✅ FIX: Remove target BEFORE feature engineering to prevent data leakage!
        target_col = "success_rate"
        if target_col in self.cleaned.columns:
            print(f"🔒 Separating target column '{target_col}' to prevent data leakage...")
            target_data = self.cleaned[target_col].copy()
            features_only = self.cleaned.drop(columns=[target_col])
        else:
            print(f"⚠️ Warning: Target column '{target_col}' not found!")
            target_data = None
            features_only = self.cleaned.copy()

        # -------------------------------
        # Experiment 8: Feature Engineering – DFS Feature Generation
        # -------------------------------

        fe = FeatureEngineer(features_only)  # ✅ Only use features, NOT target!
        self.features = fe.generate_features(max_depth=feature_depth)

        # ✅ Add target back AFTER feature engineering
        if target_data is not None:
            self.features[target_col] = target_data
            print(f"✅ Target column '{target_col}' added back after feature engineering")

        # -------------------------------
        # Experiment 9: Data Validation and Feature Filtering
        # -------------------------------

        validator = DataValidator(self.features)
        self.validation = validator.validate()

        # Remove constant and low-variance columns (but keep target)
        const_cols = self.validation[self.validation["is_constant"] == True].index.tolist()
        # ✅ Don't remove target even if constant
        const_cols = [c for c in const_cols if c != target_col]
        if const_cols:
            print(f"🗑 Removing constant columns: {const_cols}")
            self.features = self.features.drop(columns=const_cols)

        low_var_cols = self.validation[self.validation.get("low_variance") == True].index.tolist()
        # ✅ Don't remove target even if low variance
        low_var_cols = [c for c in low_var_cols if c != target_col]
        if low_var_cols:
            print(f"⚠️ Removing low variance columns: {low_var_cols}")
            self.features = self.features.drop(columns=low_var_cols)

        # Drop raw timestamp and exploded timestamp features
        self.features = self.features.drop(
            columns=[c for c in self.features.columns if c != target_col and ("timestamp" in c.lower() or "(" in c)],
            errors="ignore"
        )

        # Encode categorical features (exclude target)
        categorical_cols = self.features.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [c for c in categorical_cols if c != target_col]
        if len(categorical_cols) > 0:
            print(f"🔤 Encoding categorical columns: {list(categorical_cols)}")
            self.features = pd.get_dummies(self.features, columns=categorical_cols, drop_first=True)

        # Detect and drop high-cardinality dummy features (exclude target column)
        high_card_cols = [
            col for col in self.features.columns
            if col != target_col and self.features[col].nunique() > 50
        ]
        if high_card_cols:
            print(f"🗑 Dropping high-cardinality features: {len(high_card_cols)} columns")
            self.features = self.features.drop(columns=high_card_cols)

        # Final check: ensure all columns are numeric (except maybe target)
        non_numeric = self.features.select_dtypes(exclude=["number"]).columns
        non_numeric = [c for c in non_numeric if c != target_col]
        if len(non_numeric) > 0:
            print(f"⚠️ Warning: Non-numeric columns still present: {list(non_numeric)}")
            self.features[non_numeric] = self.features[non_numeric].astype(str)
            self.features = pd.get_dummies(self.features, columns=non_numeric, drop_first=True)

        print("✅ Pipeline completed successfully.")
        print(f"📊 Final dataset: {self.features.shape[0]} rows, {self.features.shape[1]} features")
        if target_col in self.features.columns:
            print(f"✅ Target column '{target_col}' is present in final dataset")
        else:
            print(f"⚠️ WARNING: Target column '{target_col}' is MISSING!")

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