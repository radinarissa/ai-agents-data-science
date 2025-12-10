    # -------------------------------
    # Experiment 10: Visualization of Processed Dataset
    # -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/processed_dataset.csv")

print("First 5 rows of the processed dataset:")
print("First 5 rows of the processed dataset:")
print(df.head())

print("Column information:")
print(df.info())

print("Descriptive statistics:")
print(df.head())

print("Column information:")
print(df.info())

print("Descriptive statistics:")
print(df.describe())

# Histogram of numeric features
df.hist(figsize=(10, 6), bins=10)
plt.suptitle("Distribution of Numeric Features", fontsize=14)
plt.tight_layout()
plt.savefig("experiments/results/histograms.png")
plt.show()

# Correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8, 6))
plt.imshow(numeric_df.corr(), cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation')
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=45)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
plt.title("Correlation Matrix (Numeric Features Only)")
plt.tight_layout()
plt.savefig("experiments/results/correlation_matrix.png")
plt.show()

print("✅ Visualization complete. Figures saved in experiments/results/")