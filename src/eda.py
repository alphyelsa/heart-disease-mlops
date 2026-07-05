import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load Dataset
# -------------------------
DATA_PATH = "data/raw/heart.csv"
OUTPUT_DIR = "figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# -------------------------
# Basic Information
# -------------------------
print("First 5 rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------
# Histograms
# -------------------------
df.hist(figsize=(14, 10), bins=15)

plt.suptitle("Feature Distributions")
plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/histograms.png", dpi=300)
plt.close()

# -------------------------
# Correlation Heatmap
# -------------------------
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=300)
plt.close()

# -------------------------
# Class Balance
# -------------------------
plt.figure(figsize=(6,4))

sns.countplot(
    x="target",
    data=df
)

plt.title("Class Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=300)
plt.close()

# -------------------------
# Boxplots
# -------------------------
numeric_cols = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

plt.figure(figsize=(10,6))

df[numeric_cols].boxplot()

plt.title("Boxplots of Numeric Features")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/boxplots.png", dpi=300)
plt.close()

print("\nEDA complete!")
print(f"Figures saved to '{OUTPUT_DIR}/'")