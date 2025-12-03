import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/processed_dataset.csv")

<<<<<<< HEAD
print("First 5 rows of the processed dataset:")
print(df.head())

print("Column information:")
print(df.info())

print("Descriptive statistics:")
=======
print("📊 First 5 rows of the processed dataset:")
print(df.head())

print("\nℹ️ Column information:")
print(df.info())

print("\n📈 Descriptive statistics:")
>>>>>>> cd9890f0e46783a92ea6995fb76c91f747bd3ae1
print(df.describe())

# Histogram of numeric features
df.hist(figsize=(10, 6), bins=10)
<<<<<<< HEAD
plt.suptitle("Distribution of Numeric Features", fontsize=14)
=======
plt.suptitle("📊 Distribution of Numeric Features", fontsize=14)
>>>>>>> cd9890f0e46783a92ea6995fb76c91f747bd3ae1
plt.tight_layout()
plt.savefig("results/histograms.png")
plt.show()

# Correlation matrix for numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8, 6))
plt.imshow(numeric_df.corr(), cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation')
plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=45)
plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
<<<<<<< HEAD
plt.title("Correlation Matrix (Numeric Features Only)")
=======
plt.title("🔸 Correlation Matrix (Numeric Features Only)")
>>>>>>> cd9890f0e46783a92ea6995fb76c91f747bd3ae1
plt.tight_layout()
plt.savefig("results/correlation_matrix.png")
plt.show()

