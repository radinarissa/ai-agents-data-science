import os

class DataProcessingAgent:
    def __init__(self, df):
        self.df = df
        self.cleaned = None
        self.features = None
        self.validation = None

    def run_pipeline(self, contamination=0.05, impute_strategy="mean", feature_depth=1):
        print("🔹 Starting Data Processing Pipeline...")

        from data_cleaning import DataCleaner
        from feature_engineering import FeatureEngineer
        from data_validation import DataValidator
       
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

        
        const_cols = self.validation[self.validation["is_constant"] == True].index.tolist()
        if const_cols:
            print(f"🗑 Removing constant columns: {const_cols}")
            self.features = self.features.drop(columns=const_cols)

        
        low_var_cols = self.validation[self.validation.get("low_variance") == True].index.tolist()
        if low_var_cols:
            print(f"⚠️ Removing low variance columns: {low_var_cols}")
            self.features = self.features.drop(columns=low_var_cols)

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

        print("💾 Saved processed dataset, validation report and logs to /experiments/")

        return self.features, self.validation
