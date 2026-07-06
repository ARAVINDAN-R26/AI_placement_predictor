# AI Placement Prediction — data_prep.py Overview and EDA

This README documents the `data_prep.py` script in this repository, explains the preprocessing steps and the exploratory plots generated, and shows how to run the script and reproduce results.

---

## Purpose
`data_prep.py` is an end-to-end preprocessing + EDA script that:
- Loads raw data from `placementdata.csv`.
- Cleans and prepares the data for analysis or modeling.
- Saves a cleaned dataset `placementdata_prepared.csv`.
- Produces numeric summary files and a set of diagnostic plots in the `eda_outputs/` folder.

All steps are designed to be reproducible and lightweight so the prepared file can be used by downstream modeling scripts.

---

## Dependencies
- Python 3.8+ (this workspace uses a venv)
- Packages (install into your environment):
  - pandas
  - numpy
  - matplotlib
  - seaborn



The script will write:
- `placementdata_prepared.csv` — cleaned dataset
- `eda_outputs/summary_statistics.csv` — numeric summary
- `eda_outputs/eda_summary.txt` — textual EDA summary
- PNG charts in `eda_outputs/` (see list below)

---

## Files produced
- `placementdata_prepared.csv`
- `eda_outputs/summary_statistics.csv`
- `eda_outputs/eda_summary.txt`
- Plot images (PNG) in `eda_outputs/`:
  - `hist_cgpa.png`
  - `hist_aptitude_score.png`
  - `hist_soft_skills.png`
  - `hist_ssc_marks.png`
  - `hist_hsc_marks.png`
  - `count_placement_status.png`
  - `count_extracurricular.png`
  - `count_placement_training.png`
  - `box_cgpa_by_placement.png`
  - `box_aptitude_by_placement.png`
  - `box_softskills_by_placement.png`
  - `heatmap_correlation.png`

---

## Preprocessing steps — what and why (detailed)

1. Load raw CSV (`placementdata.csv`)
   - What: Read the raw dataset into a pandas DataFrame.
   - Why: All analysis and cleaning require the data in memory as a structured table.

2. Drop duplicate rows
   - What: Remove exact duplicate rows from the dataset.
   - Why: Duplicates distort frequency counts and can bias model training and evaluation.

3. Coerce key columns to numeric using `pd.to_numeric(..., errors='coerce')`
   - What: Convert values in columns such as `CGPA`, `AptitudeTestScore`, `SoftSkillsRating`, `SSC_Marks`, `HSC_Marks` to numeric types; invalid parses become `NaN`.
   - Why: Numeric arithmetic (means, correlations, model inputs) requires numeric dtypes and coercion surfaces parse issues as missing values to be handled explicitly.

4. Impute numeric NaNs with column median
   - What: Replace missing numeric values with the median for that column.
   - Why: Median imputation is robust to outliers and preserves central tendency, making it a safe default for many numeric features.

5. Impute categorical NaNs with mode (most frequent value)
   - What: For non-numeric columns, missing values are filled with the column mode.
   - Why: Mode-based imputation is simple, keeps the most common category, and avoids introducing new synthetic categories.

6. Encode binary categorical fields (`ExtracurricularActivities`, `PlacementTraining`) to 0/1
   - What: Map `Yes`/`No` (or variants) to 1/0.
   - Why: Numeric encoding is required for statistical summaries and model inputs; binary encoding is straightforward and interpretable.

7. Encode the target `PlacementStatus` to 0/1
   - What: Map `Placed`/`NotPlaced` (and case variants) to 1/0.
   - Why: Models and evaluation metrics require numeric target labels for supervised learning.

8. Cast `StudentID` and `PlacementStatus` to integer (`Int64`) and strip whitespace from object columns
   - What: Ensure consistent types and remove stray whitespace.
   - Why: Prevents subtle type and string-matching issues later; `StudentID` is kept but should not be used as a predictor (it is an identifier).

9. Save cleaned dataset
   - What: Write `placementdata_prepared.csv`.
   - Why: Having a prepared dataset speeds iterative modeling and reporting without re-running cleaning each time.

---

## Plots produced — explanation and rationale (detailed)

Each plot was chosen to surface a particular aspect of data quality or predictive signal.

- Histogram — `CGPA` (`hist_cgpa.png`)
  - Rationale: Shows central tendency, spread, skewness, and potential multimodality among student GPAs.
  - Why important: CGPA is often a strong predictor of placement; distribution informs whether to scale, transform, or bin the feature.

- Histogram — `AptitudeTestScore` (`hist_aptitude_score.png`)
  - Rationale: Reveals score spread, densest score ranges, and potential ceiling/floor effects.
  - Why important: Aptitude test performance is frequently a direct determinant of placement outcome — understanding its distribution helps choose thresholds or transforms.

- Histogram — `SoftSkillsRating` (`hist_soft_skills.png`)
  - Rationale: Shows whether soft-skills ratings are concentrated at a few discrete values or nearly continuous.
  - Why important: If the rating has few unique values, treat it as ordinal/categorical in models; if continuous, keep numeric.

- Histograms — `SSC_Marks` and `HSC_Marks` (`hist_ssc_marks.png`, `hist_hsc_marks.png`)
  - Rationale: Visualize earlier academic records for range and cluster patterns.
  - Why important: Background academic performance can carry signal; histograms show if values need normalization or further cleaning.

- Countplot — `PlacementStatus` (`count_placement_status.png`)
  - Rationale: Visualize class balance between placed (1) and not placed (0).
  - Why important: Class imbalance affects model training and evaluation — if imbalance is significant, use stratified sampling, reweighting, or resampling.

- Countplots — `ExtracurricularActivities`, `PlacementTraining` (`count_extracurricular.png`, `count_placement_training.png`)
  - Rationale: Show prevalence of these binary features.
  - Why important: If values are too skewed (almost all 0 or 1), they may provide little predictive power alone; they may still be useful in interaction with other features.

- Boxplots by `PlacementStatus` — `CGPA`, `AptitudeTestScore`, `SoftSkillsRating` (the three `box_*_by_placement.png` files)
  - Rationale: Compare distributions between placed and not-placed groups (medians, IQRs, outliers).
  - Why important: If placed students consistently have higher medians (and limited overlap), these features are likely informative and justify stronger modeling focus.

- Correlation heatmap (`heatmap_correlation.png`) — `StudentID` excluded
  - Rationale: Pairwise Pearson correlations among numeric features and with the target.
  - Why important: Detect multicollinearity (high predictor–predictor correlations) that can affect linear models and interpretability. Identify features strongly correlated with `PlacementStatus` to prioritize feature engineering. `StudentID` is excluded because it's an identifier without predictive meaning and could distort correlation interpretation.

---

