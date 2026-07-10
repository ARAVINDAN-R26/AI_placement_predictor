## Data cleaning and preparation

## Dataset Overview

- Original dataset: placementdata_dirty.csv
- Original shape: 10,010 rows × 12 columns
- Cleaned dataset: cleaned_data.csv
- Final shape: 9,077 rows × 12 columns
- Rows removed: 933 duplicates

Columns:
- StudentID, CGPA, Internships, Projects, Workshops/Certifications
- AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities
- PlacementTraining, SSC_Marks, HSC_Marks, PlacementStatus

---

## TASK 1: Dataset Loading

The dataset was successfully loaded into a pandas DataFrame. 
The first 5 rows of the dataframe is displayed

Data Types:
- Numeric columns (float): StudentID, CGPA, Internships, Projects, Workshops/Certifications, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks

- Categorical columns (str): ExtracurricularActivities, PlacementTraining, 
  PlacementStatus

shape: 
  10,010 rows × 12 columns

**Key observations:**
- All numeric columns are stored as float
- Categorical columns contain text values with inconsistent capitalization (Yes/YES/yes)
- The dataset has 12 features

---

## TASK 2: Null Value Analysis

### Removing negative values from numeric columns

  - Since the dataset contains negative values which are never possible in those particular columns, all the negative values are getting replaced with NAN

### Null Percent

| Column | Null Count | Null % |
|--------|-----------|--------|
| StudentID | 470 | 4.70% |
| CGPA | 470 | 4.70% |
| Internships | 470 | 4.70% |
| Projects | 470 | 4.70% |
| Workshops/Certifications | 470 | 4.70% |
| AptitudeTestScore | 470 | 4.70% |
| SoftSkillsRating | 470 | 4.70% |
| ExtracurricularActivities | 501 | 5.00% |
| PlacementTraining | 501 | 5.00% |
| SSC_Marks | 470 | 4.70% |
| HSC_Marks | 470 | 4.70% |
| PlacementStatus | 501 | 5.00% |

There is no columns exceeding the NULL% of 20. So the NULL values in the columns are filled with the median.

**Why use median:**
  The data is skewed, so filling it with 'mean' still retians the skewness of the data. Mean will not solve the outlier issue.
  So we are using median to fill the missing values as they are more robust to outliers.

  The categorical columns are filled with the most repeated values as it doesnt do much changes to the original data.

**Result:** Duplicated values are removed and the NULL values are handled. Now the dataset does not contain any NULL or duplicates.
But the best method for this dataset would be to remove the rows since these values for the dataset cannot be adjusted with mean or median as it would lead to wrong data. 

---

## TASK 3: Duplicate Detection and Removal

- **Duplicates found:** 933 rows (9.32% of original data) and all 933 are removed.
- **Rows after removal:** 9,077 (90.68% of original data)

## NULL% after removing duplicates
StudentID: 0.00%
CGPA: 0.00%
Internships: 0.00%
Projects: 0.00%
Workshops/Certifications: 0.00%
AptitudeTestScore: 0.00%
SoftSkillsRating: 0.00%
ExtracurricularActivities: 0.00%
PlacementTraining: 0.00%
SSC_Marks: 0.00%
HSC_Marks: 0.00%
PlacementStatus: 0.00%


- **Impact on null percentages:** Minimal. The NULL values are already cleared in the TASK 2, so the removal of duplicates doesnt change.

**Conclusion:** Removing duplicates significantly cleaned the dataset without introducing new null values or biasing any column's missingness pattern.

---

## TASK 4: Data Type Correction

- converted the categorical columns 'ExtracurricularActivities','PlacementTraining', 'PlacementStatus' as string type, and converted the column values to lowercase since the strings are inconsistent.

- converted 'PlacementStatus' columns into a category dtype. This reduces memory requirement for repeatative categorical values and speeds up the operation.

### Memory Optimization

| Metric | Before | After | Saved |
|--------|--------|-------|-------|
| Total Memory |2181257 Bytes | 1672963 Bytes | 508294 Bytes |


---

## TASK 5: Descriptive Statistics & Skewness

**Positive skewness vs Negative skewness**
- A few very large values stretch the distribution to the right is positive skew
- A few very large values stretch the distribution to the left is negative skew

- The most skewed column in the dataset is CGPA
- Most students have CGPA around 7-8, but a few have unrealistic values (77, 50025)
- This lead to positive skewness (right skewed) which stretches the distribution to the right.
 
**Implications for imputation:**
- When imputing Null with mean values the problem the average leans towards the outlier data. 
- This makes the data skewed again. It doesnt solve the issue
- To overcome this median which is the middle value is taken. since it is more robust towards outliers

---

## TASK 6: Outlier Detection with IQR

### Outlier Summary (IQR Method: Q1 − 1.5×IQR, Q3 + 1.5×IQR)

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | % |
|--------|----|----|-----|-------------|-------------|---------------|---|
| StudentID | 2505 | 7515 | 5010 | -5010 | 15030 | 29 | 0.32% |
| CGPA | 7.4 | 8.2 | 0.8 | 6.2 | 9.4 | 29 | 0.32% |
| Internships | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 4059 | 44.72% |
| Projects | 1.0 | 3.0 | 2.0 | -2.0 | 6.0 | 14 | 0.15% |
| Workshops/Certifications | 0.0 | 2.0 | 2.0 | -3.0 | 5.0 | 14 | 0.15% |
| AptitudeTestScore | 73 | 87 | 14 | 52 | 108 | 29 | 0.32% |
| SoftSkillsRating | 4.0 | 4.7 | 0.7 | 2.95 | 5.75 | 29 | 0.32% |
| SSC_Marks | 59 | 78 | 19 | 30.5 | 106.5 | 29 | 0.32% |
| HSC_Marks | 67 | 83 | 16 | 43 | 107 | 29 | 0.32% |

### Analysis & Handling Strategy

**1. Internships (44.72% outliers):**
- In the dataset 'Internship' has the highest outlier count.
- But the outliers are expected for values with low variance. 
- The percentile of the data is pointing to 1 internship which most of the student has done, the outlier occurs even if a student does one extra internship
- Decision: This is still within the expected category, so we will retain the data.

**2. CGPA, AptitudeTestScore, SoftSkillsRating, SSC_Marks, HSC_Marks (~0.32% outliers):**
- Outlier count is appreoximately 29 rows and it is less than 1% of data.
- This is likely to be caused because of wrong data entry.
- Decision: Retain the values but flag it for now. We will cap the data if there is any issue.

**3. StudentID :**
- Only 29 outliers, represents valid student IDs
- Decision: This column can be ignored as it is not necessay for maodelling. This feature will be ignored.

**4. Projects, Workshops (0.15% outliers):**
- Very small outlier count, It is within the expected range
- Decision: Retain it.

**Outlier Analysis Output:**
- The analysis for the outlier is stored in eda_outputs/outlier_analysis.csv

---

## TASK 7: Visualizations

All visualizations are saved in the `eda_outputs/` directory.

### 1. Line Plot: CGPA Over Sample Index
- Shows CGPA values for the first sorted 100 records in sequence
- Reveals in first 100 records almost everyone scores the same CGPA


### 2. Bar Chart: Average CGPA by Placement Status
- Placed students: Mean CGPA = 8.09
- Not Placed students:** Mean CGPA = 7.56
- **Difference:** 0.53 CGPA points (about 7% higher for placed students)
- **Insight:** CGPA is predictive of placement outcome; higher achievers are more likely to be placed

### 3. Histogram: Most Skewed Column (CGPA)
- **Shape:** Highly right-skewed with a long tail
- **Distribution:** Most values cluster between 7-8.5
- **Outliers:** Clear spike at extreme values (>10), confirming data quality issues
- **Implication:** The 44% outliers in low-variance columns (Internships) are data artifacts or valid sparse categories that require handling before modeling

### 4. Scatter Plot: CGPA vs Aptitude Test Score
- shows very high outlier present in CGPA and Aptitude test score
- But the remaining data shows the CGPA and Aptitude test scores are in the same dimension

### 5. Box Plot: CGPA by Placement Status
- **Placed students:** Higher median (~8.1), smaller spread
- **Not Placed students:** Lower median (~7.6), larger spread
- **Insight:** Placed students have more consistent (less variable) CGPA; better students cluster in placed category

---

## TASK 8: Correlation Heatmap

### Key Finding: Highest Correlation
**AptitudeTestScore ↔ SoftSkillsRating: r = 0.96**
**AptitudeTestScore ↔ CGPA: r = 0.96**

This extremely high correlation suggests these variables measure nearly identical phenomena. Possible explanations:

1. **Data Entry Error:** Same data copied to both columns
2. **Scaling Issue:** Both scaled identically due to measurement error
3. **Genuine Overlap:** Both truly capture "academic and professional aptitude"
4. **Redundancy:** One variable could be derived from the other

---

## TASK 9a: Imputation Strategy Comparison

### Two Most Skewed Columns

**1. CGPA (Skewness = 21.51)**
- Mean: 7.7803
- Median: 7.7000
- **Why Median?** The column is positively skewed, so extreme high values pull the mean upward. The median is more representative of the typical student.

**2. SoftSkillsRating (Skewness = 21.14)**
- Mean: 4.3707
- Median: 4.4000
- **Why Median?** The column is also positively skewed, so the median is a better choice than the mean for imputation.

### Why use Median

**Justification:**
1. The two most skewed numeric columns are positively skewed.
2. In a positively skewed column, the mean is pulled upward by extreme high values.
3. The median is more stable and better represents the center of the data.
4. The median was applied using fillna() to any remaining nulls in these columns.

**Result:** Any remaining nulls in these two columns were filled with the median, and the null check confirmed that no nulls remain.

---

## TASK 9b: Spearman Rank Correlation

## Method Overview

Spearman Rank Correlation is used to measure the relationship between two variables based on their **rank** instead of their actual values.

It works as follows:

1. Convert all values into ranks.
2. Calculate the correlation using these ranks.
3. The result shows whether the variables increase or decrease together, even if the relationship is not perfectly linear.

### Why compare Pearson and Spearman Correlation?

* **Pearson Correlation** measures only **linear relationships** between two variables.
* **Spearman Correlation** measures **monotonic relationships**, where one variable consistently increases or decreases with the other, even if the relationship is not linear.

By comparing both values, we can understand the nature of the relationship.

* If **Pearson and Spearman values are almost equal**, the relationship is approximately linear.
* If **Spearman is much higher than Pearson**, the relationship is monotonic but non-linear.
* If **Pearson is much higher than Spearman**, it usually indicates that outliers are influencing the Pearson correlation.

---

## Top 3 Pairs with the Largest Pearson-Spearman Difference

| Pair                         | Pearson | Spearman | Difference | Interpretation                                                                                                                                                                                                                     |
| ---------------------------- | ------- | -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CGPA ↔ SoftSkillsRating      | 0.9653  | 0.4236   | 0.5418     | Pearson shows a very strong correlation, but Spearman is much lower. This indicates that a few extreme values are increasing the Pearson correlation. The actual relationship between these features is only moderately monotonic. |
| CGPA ↔ SSC_Marks             | 0.9360  | 0.4215   | 0.5145     | Similar to the previous pair, the large difference suggests that outliers are affecting the Pearson correlation, making the relationship appear stronger than it actually is.                                                      |
| StudentID ↔ SoftSkillsRating | 0.5161  | 0.0135   | 0.5026     | Spearman correlation is almost zero, which means there is practically no relationship between these variables. The Pearson correlation is mainly influenced by extreme values.                                                     |

---

## Correlation Method to Use for Feature Selection

For this dataset, **Spearman Rank Correlation** is a better choice for feature selection.

### Reasons

1. Spearman is **less affected by outliers** because it uses the rank of the values instead of the actual values.

2. This dataset contains several extreme values, such as negative and unusually large CGPA values and Aptitude Test Scores. These outliers can artificially increase the Pearson correlation.

3. The large differences between Pearson and Spearman correlations for several feature pairs indicate that Pearson is being influenced by outliers rather than the true relationship between the variables.

4. During data preprocessing in Part 2, these outliers will be removed or handled appropriately. Therefore, Spearman correlation provides a more reliable measure of the relationship between features in the current dataset.


---

## TASK 9c: Grouped Aggregation

## CGPA by Placement Status

| Placement Status | Mean CGPA | Standard Deviation | Count |
| ---------------- | --------: | -----------------: | ----: |
| Not Placed       |    7.5592 |             2.9051 |  5011 |
| Placed           |    8.0868 |             2.7027 |  3606 |

## Key Insights

### 1. Which group has the higher average CGPA?

The **placed** students have a higher average CGPA (**8.0868**) compared to **not placed** students (**7.5592**).

The difference between the two groups is approximately **0.53 CGPA points**, which is around **7% higher**. This indicates that students with a higher CGPA are generally more likely to get placed.

---

### 2. Which group has more variation in CGPA?

The **not placed** group has a higher standard deviation (**2.9051**) than the **placed** group (**2.7027**).

This means the CGPA values of not placed students are more spread out. Some students have very low CGPA, while others have relatively high CGPA.

The higher variation suggests that **CGPA alone is not enough to predict whether a student will be placed**.

For example:

* Some students with a high CGPA are still not placed.
* Some students with a lower CGPA are placed.

This means other factors, such as **Aptitude Test Score, Soft Skills Rating, Projects, Internship Experience, and Technical Skills**, should also be considered while building the prediction model.

---

### 4. Does CGPA have predictive power?

Yes. The average CGPA of placed students is higher than that of not placed students, which shows that CGPA has a relationship with placement.

However, the difference is not very large. Therefore, **CGPA should not be used as the only feature for prediction**. It should be combined with other important features to build a better machine learning model.

---

## TASK 10: Cleaned Dataset

The cleaned file is saved into cleaned_data.csv file which will be used for part
---
