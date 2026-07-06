from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input_file = Path("placementdata.csv")
output_file = Path("placementdata_prepared.csv")
output_dir = Path("eda_outputs")
output_dir.mkdir(exist_ok=True)


def encode_yes_no(series):
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    return series.astype(str).str.strip().map(mapping).astype("Int64")


def encode_target(series):
    mapping = {"Placed": 1, "NotPlaced": 0, "placed": 1, "notplaced": 0}
    return series.astype(str).str.strip().map(mapping).astype("Int64")


def plot_histogram(df, column, output_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=20, color="steelblue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_countplot(df, column, output_path):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, color="mediumseagreen")
    plt.title(f"Count of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_boxplot(df, x_col, y_col, output_path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=x_col, y=y_col, color="lightblue")
    plt.title(f"{y_col} by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_eda(df):
    print("\n=== Exploratory Data Analysis ===")
    print("Dataset shape:", df.shape)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_rows = []
    for col in numeric_columns:
        s = df[col]
        summary_rows.append(
            {
                "Column": col,
                "Count": int(s.count()),
                "Mean": round(float(s.mean()), 4),
                "Median": round(float(s.median()), 4),
                "Std": round(float(s.std()), 4),
                "Min": round(float(s.min()), 4),
                "Max": round(float(s.max()), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)

    with open(output_dir / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("EDA Summary\n")
        f.write("==========\n")
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        f.write(summary_df.to_string(index=False))

    print("\nGenerating plots...")
    plot_histogram(df, "CGPA", output_dir / "hist_cgpa.png")
    plot_histogram(df, "AptitudeTestScore", output_dir / "hist_aptitude_score.png")
    plot_histogram(df, "SoftSkillsRating", output_dir / "hist_soft_skills.png")
    plot_histogram(df, "SSC_Marks", output_dir / "hist_ssc_marks.png")
    plot_histogram(df, "HSC_Marks", output_dir / "hist_hsc_marks.png")

    plot_countplot(df, "PlacementStatus", output_dir / "count_placement_status.png")
    plot_countplot(df, "ExtracurricularActivities", output_dir / "count_extracurricular.png")
    plot_countplot(df, "PlacementTraining", output_dir / "count_placement_training.png")

    plot_boxplot(df, "PlacementStatus", "CGPA", output_dir / "box_cgpa_by_placement.png")
    plot_boxplot(df, "PlacementStatus", "AptitudeTestScore", output_dir / "box_aptitude_by_placement.png")
    plot_boxplot(df, "PlacementStatus", "SoftSkillsRating", output_dir / "box_softskills_by_placement.png")

    plt.figure(figsize=(10, 8))
    corr_columns = [col for col in numeric_columns if col != "StudentID"]
    corr = df[corr_columns].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_correlation.png", dpi=300)
    plt.close()

    print("EDA outputs saved to:", output_dir)
    print("Plots stored as png")
    # for path in sorted(output_dir.glob("*")):
    #     print(f"- {path.name}")


def main():
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(input_file)
    changes = []

    print("Dataset file:", input_file)
    print("Total rows:", df.shape[0])
    print("Total columns:", df.shape[1])
    print("Column names:", ", ".join(df.columns.tolist()))

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates()
        changes.append(f"Removed {duplicate_count} duplicate row(s).")
    else:
        changes.append("No duplicate rows found.")

    numeric_columns = [
        "StudentID",
        "CGPA",
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "SSC_Marks",
        "HSC_Marks",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_filled_columns = []
    for col in df.select_dtypes(exclude="number").columns:
        if df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])
                categorical_filled_columns.append(col)

    if categorical_filled_columns:
        changes.append(
            "Filled missing categorical values with the mode for: " + ", ".join(categorical_filled_columns)
        )

    for col in ["ExtracurricularActivities", "PlacementTraining"]:
        if col in df.columns:
            df[col] = encode_yes_no(df[col])

    if "PlacementStatus" in df.columns:
        df["PlacementStatus"] = encode_target(df["PlacementStatus"])

    if "StudentID" in df.columns:
        df["StudentID"] = df["StudentID"].astype("Int64")
    if "PlacementStatus" in df.columns:
        df["PlacementStatus"] = df["PlacementStatus"].astype("Int64")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    print("\nDataset shape after cleaning:")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nData types after cleaning:")
    print(df.dtypes)

    print("\nChanges made during preprocessing:")
    for change in changes:
        print(f"- {change}")

    df.to_csv(output_file, index=False)
    print(f"\nPrepared data saved to: {output_file}")

    run_eda(df)


main()
