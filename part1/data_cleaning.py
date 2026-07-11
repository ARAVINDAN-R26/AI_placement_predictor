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



# TASK 1: Load the dataset and inspect it
# ============================================================================
print("\nTASK 1: LOAD AND INSPECT DATASET")

df = pd.read_csv(input_file)

print("\n First 5 Rows")
print(df.head())

print("\nData Types")
print(df.dtypes)

print("\n Dataset Shape ")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")



# TASK 2: Null value analysis
# ============================================================================
print("\nTASK 2: NULL VALUE ANALYSIS")

# Check numeric columns for negative values
for column in df.columns:
    if df[column].dtype != "object":
        negative_count = (df[column] < 0).sum()
        if negative_count > 0:
            df[column] = df[column].where(df[column] >= 0, np.nan)


# Null values before removing duplicates
null_count = df.isnull().sum()
null_percent = (df.isnull().sum() / df.shape[0]) * 100

print("\n Null Values Summary ")
for col in df.columns:
    print(f"{col}: null count = {null_count[col]}, null percentage = {null_percent[col]:.2f}%")


# Check columns with more than 20% nulls after removing duplicates
print("\n Columns Exceeding 20% Null Rate")
high_null_found = False
for col in df.columns:
    if null_percent[col] > 20:
        print(f"{col}: {null_percent[col]:.2f}%")
        high_null_found = True

if not high_null_found:
    print("No columns exceeded 20%\n")


# Keep a copy before imputation so Task 9a can compare mean and median before filling
df_before_imputation = df.copy()

# Fill numeric columns with median
for column in df.columns:
    if df[column].dtype != "object":
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            # print(f"Filled missing values in '{column}' with median: {median_value}")

# Fill categorical columns with the most repeated value
for column in df.columns:
    if df[column].dtype == "object":
        if df[column].isnull().any():
            mode_value = df[column].mode()
            if not mode_value.empty:
                fill_value = mode_value[0]
                df[column] = df[column].fillna(fill_value)
                # print(f"Filled missing values in '{column}' with mode: {fill_value}")




# TASK 3: Duplicate detection and removal
# ============================================================================
print("\nTASK 3: DUPLICATE DETECTION AND REMOVAL")

# Remove duplicates and check null percentage again
print("\n Removing Duplicates")
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate row(s).")
else:
    print("No duplicates found.")

# Null percentage after removing duplicates
null_percent_after = (df.isnull().sum() / df.shape[0]) * 100

print("\n Null Percentage After Removing Duplicates ")
for col in df.columns:
    print(f"{col}: {null_percent_after[col]:.2f}%")


# TASK 4: Data type correction
# ============================================================================
print("\nTASK 4: DATA TYPE CORRECTION")

print("\n--- Memory Usage Before Conversion ---")
memory_before = df.memory_usage(deep=True).sum()
print(f"Total memory: {memory_before:} Bytes")

# Standardize case in categorical columns
categorical_cols = ["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Keep as object for clarity, though category would save memory

print(f"\nStandardized text in categorical columns: {categorical_cols}")

# Convert appropriate string columns to category dtype for memory efficiency
df["PlacementStatus"] = df["PlacementStatus"].astype("category")
print("\nConverted 'PlacementStatus' to category dtype.")

print("\n--- Memory Usage After Conversion ---")
memory_after = df.memory_usage(deep=True).sum()
print(f"Total memory: {memory_after:} Bytes")
print(f"Memory saved: {memory_before - memory_after:} Bytes")


# TASK 5: Descriptive statistics and skewness
# ============================================================================
print("\nTASK 5: DESCRIPTIVE STATISTICS AND SKEWNESS")

numeric_df = df.select_dtypes(include=[np.number])

print("\n--- Descriptive Statistics ---")
print(numeric_df.describe())

skewness_data = []
for col in numeric_df.columns:
    skew_val = numeric_df[col].skew()
    skewness_data.append({"Column": col, "Skewness": skew_val})
    
skewness_df = pd.DataFrame(skewness_data).sort_values("Skewness", key=abs, ascending=False)

print("\n--- Skewness Analysis ---")
print(skewness_df.to_string(index=False))

most_skewed_col = skewness_df.iloc[0]["Column"]
most_skewed_val = skewness_df.iloc[0]["Skewness"]
print(f"\nMost skewed column: '{most_skewed_col}' skewness = {most_skewed_val:.4f}")


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
        # "Outlier %": (outlier_count / len(numeric_df)) * 100
    })
    
    print(f"\n{col}:")
    print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers found: {outlier_count} ({(outlier_count/len(numeric_df))*100:.2f}%)")

outlier_df = pd.DataFrame(outlier_info)
outlier_df.to_csv(output_dir / "outlier_analysis.csv", index=False)
print("\nOutlier analysis saved to 'outlier_analysis.csv'")


# TASK 7: Visualizations (5 types)
# ============================================================================
print("\nTASK 7: VISUALIZATIONS")

# 1. Line plot
plt.figure(figsize=(12, 5))
plt.plot(numeric_df["CGPA"].sort_values().head(100), marker="o", markersize=3, linewidth=1)
plt.title("CGPA Over Sample Index (First 100 Records)")
plt.xlabel("Sample Index")
plt.ylabel("CGPA")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "01_line_plot_cgpa.png", dpi=300)
plt.close()
print("Line plot saved: 01_line_plot_cgpa.png")

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
print("Bar chart saved: 02_bar_chart_cgpa_by_placement.png")

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
print(f"Histogram saved: 03_histogram_most_skewed.png ({most_skewed_col})")

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
print("Scatter plot saved: 04_scatter_cgpa_vs_aptitude.png")

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
print("Box plot saved: 05_boxplot_cgpa_by_placement.png")


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
print("Correlation heatmap saved: correlation_heatmap_pearson.png")


# ============================================================================
# TASK 9a: Imputation strategy comparison
# ============================================================================
print("\nTASK 9a: IMPUTATION STRATEGY COMPARISON")

# Get top 2 most skewed columns
top_2_skewed = skewness_df.head(2)["Column"].tolist()

print("\n--- Mean vs Median for Most Skewed Columns ---")
for col in top_2_skewed:
    mean_val = df_before_imputation[col].mean()
    median_val = df_before_imputation[col].median()
    skew_val = skewness_df.loc[skewness_df["Column"] == col, "Skewness"].iloc[0]
    
    print(f"\n{col}:")
    print(f"  Mean:   {mean_val:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  Skewness: {skew_val:.4f}")


    if df[col].isnull().any():
        df[col] = df[col].fillna(median_val)
        print(f"  Filled remaining nulls in '{col}' with median using fillna().")
    else:
        print(f"  No remaining nulls in '{col}'.")

    print(f"  Null count after fillna: {df[col].isnull().sum()}")

print("\nChosen imputation strategy: MEDIAN")
print("Reason: for skewed columns, the median is more representative than the mean because extreme values pull the mean away from the center.")


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
print("\norrelation comparison saved to 'correlation_comparison.csv'")



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

print(f"\nGroup with highest mean: {highest_mean_group}")
print(f"Group with highest std dev: {highest_std_group}")
print(f"Ratio of highest to lowest mean: {mean_ratio:.4f}")

grouped_agg.to_csv(output_dir / "grouped_aggregation.csv")
print("\nGrouped aggregation saved to 'grouped_aggregation.csv'")


# TASK 10: Save cleaned dataset
# ============================================================================
print("\nTASK 10: SAVE CLEANED DATASET")

# Convert PlacementStatus back to string for saving
df["PlacementStatus"] = df["PlacementStatus"].astype(str)

df.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to: {output_file}")
print(f"Final dataset shape: {df.shape}")

print("DATA CLEANING AND EDA COMPLETE")
print(f"\nAll outputs saved to: {output_dir}")
