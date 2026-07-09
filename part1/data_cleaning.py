"""
Data Cleaning and Exploratory Data Analysis (EDA) Script
========================================================
This script performs comprehensive data cleaning and analysis on the placement dataset.
All \nTASKs are designed to be beginner-friendly with clear comments.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
script_folder = Path(__file__).parent
input_file = script_folder / "placementdata_dirty.csv"
output_file = script_folder / "cleaned_data.csv"
output_dir = script_folder / "eda_outputs"
output_dir.mkdir(exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ============================================================================
# TASK 1: Load the dataset and inspect it
# ============================================================================
print("\nTASK 1: LOAD AND INSPECT DATASET")

df = pd.read_csv(input_file)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# ============================================================================
# TASK 2: Null value analysis
# ============================================================================
print("\nTASK 2: NULL VALUE ANALYSIS")

null_count = df.isnull().sum()
null_percent = (df.isnull().sum() / df.shape[0]) * 100

null_info = pd.DataFrame({
    "Column": df.columns,
    "Null Count": null_count.values,
    "Null Percentage": null_percent.values
})

print("\n--- Null Values Summary ---")
print(null_info.to_string(index=False))

# Identify columns with >20% nulls
high_null_cols = null_info[null_info["Null Percentage"] > 20]["Column"].tolist()
if high_null_cols:
    print(f"\n[WARNING] Columns exceeding 20% null rate: {high_null_cols}")
else:
    print("\n[OK] No columns exceed 20% null rate.")

# For columns below 20% nulls, fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any() and null_percent[col] < 20:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled '{col}' with median: {median_val}")

print("\n[OK] Numeric columns below 20% nulls filled with median.")


# ============================================================================
# TASK 3: Duplicate detection and removal
# ============================================================================
print("\nTASK 3: DUPLICATE DETECTION AND REMOVAL")

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"[OK] Removed {duplicate_count} duplicate row(s).")
    print(f"New dataset shape: {df.shape}")
else:
    print("[OK] No duplicates found.")


# ============================================================================
# TASK 4: Data type correction
# ============================================================================
print("\nTASK 4: DATA TYPE CORRECTION")

print("\n--- Memory Usage Before Conversion ---")
memory_before = df.memory_usage(deep=True).sum() / 1024 ** 2
print(f"Total memory: {memory_before:.2f} MB")

# Standardize case in categorical columns
categorical_cols = ["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Keep as object for clarity, though category would save memory

print(f"\n[OK] Standardized text in categorical columns: {categorical_cols}")

# Convert appropriate string columns to category dtype for memory efficiency
df["PlacementStatus"] = df["PlacementStatus"].astype("category")
print("[OK] Converted 'PlacementStatus' to category dtype (memory efficient).")

print("\n--- Memory Usage After Conversion ---")
memory_after = df.memory_usage(deep=True).sum() / 1024 ** 2
print(f"Total memory: {memory_after:.2f} MB")
print(f"Memory saved: {memory_before - memory_after:.4f} MB")


# ============================================================================
# TASK 5: Descriptive statistics and skewness
# ============================================================================
print("\nTASK 5: DESCRIPTIVE STATISTICS AND SKEWNESS")

numeric_df = df.select_dtypes(include=[np.number])

print("\n--- Descriptive Statistics ---")
print(numeric_df.describe().to_string())

skewness_data = []
for col in numeric_df.columns:
    skew_val = numeric_df[col].skew()
    skewness_data.append({"Column": col, "Skewness": skew_val})
    
skewness_df = pd.DataFrame(skewness_data).sort_values("Skewness", key=abs, ascending=False)

print("\n--- Skewness Analysis ---")
print(skewness_df.to_string(index=False))

most_skewed_col = skewness_df.iloc[0]["Column"]
most_skewed_val = skewness_df.iloc[0]["Skewness"]
print(f"\n[OK] Most skewed column: '{most_skewed_col}' (skewness = {most_skewed_val:.4f})")


# ============================================================================
# TASK 6: Outlier detection with IQR
# ============================================================================
print("\nTASK 6: OUTLIER DETECTION WITH IQR")

outlier_info = []
for col in numeric_df.columns:
    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
    outlier_count = outlier_mask.sum()
    
    outlier_info.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count,
        "Outlier %": (outlier_count / len(numeric_df)) * 100
    })
    
    print(f"\n{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers found: {outlier_count} ({(outlier_count/len(numeric_df))*100:.2f}%)")

outlier_df = pd.DataFrame(outlier_info)
outlier_df.to_csv(output_dir / "outlier_analysis.csv", index=False)
print("\n[OK] Outlier analysis saved to 'outlier_analysis.csv'")


# ============================================================================
# TASK 7: Visualizations (5 types)
# ============================================================================
print("\nTASK 7: VISUALIZATIONS")

# 1. Line plot
plt.figure(figsize=(12, 5))
plt.plot(numeric_df["CGPA"].head(100), marker="o", markersize=3, linewidth=1)
plt.title("CGPA Over Sample Index (First 100 Records)")
plt.xlabel("Sample Index")
plt.ylabel("CGPA")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "01_line_plot_cgpa.png", dpi=300)
plt.close()
print("[OK] Line plot saved: 01_line_plot_cgpa.png")

# 2. Bar chart - Mean of numeric column by categorical column
plt.figure(figsize=(10, 5))
groupby_data = df.groupby("PlacementStatus")["CGPA"].mean()
groupby_data.plot(kind="bar", color=["#FF6B6B", "#4ECDC4"])
plt.title("Average CGPA by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Average CGPA")
plt.xticks(rotation=0)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(output_dir / "02_bar_chart_cgpa_by_placement.png", dpi=300)
plt.close()
print("[OK] Bar chart saved: 02_bar_chart_cgpa_by_placement.png")

# 3. Histogram - Most skewed column
plt.figure(figsize=(10, 5))
sns.histplot(data=numeric_df, x=most_skewed_col, bins=20, kde=True, color="steelblue")
plt.title(f"Distribution of {most_skewed_col} (Most Skewed Column)")
plt.xlabel(most_skewed_col)
plt.ylabel("Frequency")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(output_dir / "03_histogram_most_skewed.png", dpi=300)
plt.close()
print(f"[OK] Histogram saved: 03_histogram_most_skewed.png ({most_skewed_col})")

# 4. Scatter plot - Expected correlation
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="CGPA", y="AptitudeTestScore", alpha=0.6, s=50)
plt.title("Relationship: CGPA vs Aptitude Test Score")
plt.xlabel("CGPA")
plt.ylabel("Aptitude Test Score")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "04_scatter_cgpa_vs_aptitude.png", dpi=300)
plt.close()
print("[OK] Scatter plot saved: 04_scatter_cgpa_vs_aptitude.png")

# 5. Box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="PlacementStatus", y="CGPA", hue="PlacementStatus", palette="Set2", legend=False)
plt.title("CGPA Distribution by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(output_dir / "05_boxplot_cgpa_by_placement.png", dpi=300)
plt.close()
print("[OK] Box plot saved: 05_boxplot_cgpa_by_placement.png")


# ============================================================================
# TASK 8: Correlation heatmap
# ============================================================================
print("\nTASK 8: CORRELATION HEATMAP")

correlation_matrix = numeric_df.corr()

# Find the pair with highest absolute correlation (excluding diagonal)
corr_abs = correlation_matrix.abs().to_numpy().copy()  # Make a writable numpy copy
np.fill_diagonal(corr_abs, 0)  # Ignore self-correlation
max_corr_idx = np.unravel_index(corr_abs.argmax(), corr_abs.shape)
col1, col2 = correlation_matrix.index[max_corr_idx[0]], correlation_matrix.columns[max_corr_idx[1]]
max_corr_value = correlation_matrix.iloc[max_corr_idx]

print(f"\nHighest absolute correlation: {col1} <-> {col2}")
print(f"Correlation coefficient: {max_corr_value:.4f}")

# Visualize
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix - All Numeric Columns")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap_pearson.png", dpi=300)
plt.close()
print("[OK] Correlation heatmap saved: correlation_heatmap_pearson.png")


# ============================================================================
# TASK 9a: Imputation strategy comparison
# ============================================================================
print("\nTASK 9a: IMPUTATION STRATEGY COMPARISON")

# Get top 2 most skewed columns
top_2_skewed = skewness_df.head(2)["Column"].tolist()

print("\n--- Mean vs Median for Most Skewed Columns ---")
imputation_stats = []
for col in top_2_skewed:
    mean_val = df[col].mean()
    median_val = df[col].median()
    skew_val = df[col].skew()
    
    print(f"\n{col}:")
    print(f"  Mean:   {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Skewness: {skew_val:.4f}")
    
    imputation_stats.append({
        "Column": col,
        "Mean": mean_val,
        "Median": median_val,
        "Skewness": skew_val
    })

print("\n[OK] Using MEDIAN for imputation (more robust to skew)")


# ============================================================================
# TASK 9b: Spearman rank correlation
# ============================================================================
print("\nTASK 9b: SPEARMAN RANK CORRELATION")

spearman_matrix = numeric_df.corr(method="spearman")

print("\n--- Spearman Correlation Matrix ---")
print(spearman_matrix.to_string())

# Compute differences
diff_matrix = (spearman_matrix - correlation_matrix).abs().to_numpy().copy()  # Make a writable numpy copy
np.fill_diagonal(diff_matrix, 0)

# Find top 3 pairs with largest differences
diff_values = []
for i in range(len(correlation_matrix)):
    for j in range(i+1, len(correlation_matrix)):
        diff_values.append({
            "Pair": f"{correlation_matrix.index[i]} <-> {correlation_matrix.columns[j]}",
            "Pearson": correlation_matrix.iloc[i, j],
            "Spearman": spearman_matrix.iloc[i, j],
            "Absolute Difference": diff_matrix[i, j]
        })

diff_df = pd.DataFrame(diff_values).sort_values("Absolute Difference", ascending=False).head(3)

print("\n--- Top 3 Pairs with Largest Pearson-Spearman Differences ---")
print(diff_df.to_string(index=False))

diff_df.to_csv(output_dir / "correlation_comparison.csv", index=False)
print("\n[OK] Correlation comparison saved to 'correlation_comparison.csv'")


# ============================================================================
# TASK 9c: Grouped aggregation
# ============================================================================
print("\nTASK 9c: GROUPED AGGREGATION")

# Group by PlacementStatus and analyze numeric columns
grouped_agg = df.groupby("PlacementStatus")["CGPA"].agg(["mean", "std", "count"])

print("\n--- Grouped Statistics: CGPA by Placement Status ---")
print(grouped_agg.to_string())

highest_mean_group = grouped_agg["mean"].idxmax()
highest_std_group = grouped_agg["std"].idxmax()
mean_ratio = grouped_agg["mean"].max() / grouped_agg["mean"].min()

print(f"\n[OK] Group with highest mean: {highest_mean_group}")
print(f"[OK] Group with highest std dev: {highest_std_group}")
print(f"[OK] Ratio of highest to lowest mean: {mean_ratio:.4f}")

grouped_agg.to_csv(output_dir / "grouped_aggregation.csv")
print("\n[OK] Grouped aggregation saved to 'grouped_aggregation.csv'")


# ============================================================================
# TASK 10: Save cleaned dataset
# ============================================================================
print("\nTASK 10: SAVE CLEANED DATASET")

# Convert PlacementStatus back to string for saving
df["PlacementStatus"] = df["PlacementStatus"].astype(str)

df.to_csv(output_file, index=False)
print(f"[OK] Cleaned dataset saved to: {output_file}")
print(f"Final dataset shape: {df.shape}")

print("\n" + "=" * 70)
print("DATA CLEANING AND EDA COMPLETE")
print("=" * 70)
print(f"\nAll outputs saved to: {output_dir}")
