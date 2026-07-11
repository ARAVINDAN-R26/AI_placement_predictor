### Supervised Machine Learning model

## TASK 1
-   The cleaned dataset from part1 is loaded.
-   After loading the data the 'CGPA' (continuous numerical value) column is taken as target for the regression model and the 'PlacementStatus' (binary column) is taken as the target for the classification model.
-   The targets 'CGPA' and "PlacementStatus' is removed and the remaining columns are considered as the features.

## TASK 2
- The categorical columns ('ExtracurricularActivities', 'PlacementTraining') are collected. After collecting the categorical columns, one-hot-encoding is performed on those columns.
- one-hot-encoding is performed on these data because these columns doesnt have any natural order since they are an yes/no columns.
- When performing onehot encoding the first dummy column is droped to avoid multicollinearity.

**why one-hot encoding avoids the false-ordinal-relationship problem?**
- One-hot encoding avoids the false ordinal relationship by representing each category as a separate binary (0 or 1)
column instead of assigning numerical values. 
- Since each category is treated independently, the model does not assume that one category is greater than, less than, or closer to another. 
- This prevents the model from learning relationships that do not actually exist between the categorical values.

## TASK 3
- The datas are splitted into training data and testing data. 
- The train and test datas are splitted for both regression features and the classification features.
- Then scaling is done based on the x_train, after that x_train and x_test are transformed.

**Why should we not fit the data on the full dataset?**
- The scaler should be fitted only on the training data. Fitting the scaler on the entire dataset would introduce data leakage because the scaling process would use the mean and standard deviation of both the training and test data.
- Because of that, information from the test set would be indirectly available during training, leading to overly optimistic model performance which doesnt represent how model performs on unseen data.

## TASK 4
- Selected the Linear regression model to predict continous output as a result.
- After doing the prediction, calculated the mean_squared_err, r2_score for the model
- Then the coefficients for each and every column is printed

Linear Regression coefficients:
StudentID                        0.035812
Internships                      0.001652
Projects                        -0.047238
Workshops/Certifications        -0.016712
AptitudeTestScore                0.810591
SoftSkillsRating                 0.766681
SSC_Marks                        0.238319
HSC_Marks                        0.584233
ExtracurricularActivities_yes   -0.078650
PlacementTraining_yes           -0.030723

**What large positive and negative coefficients indicate?**
A large positive coefficient indicates that the particular feature has more impact on the prediction. As one feature value increases other value increases to the magnitude of that coefficient.

A large negative coefficient indicates that as the value of the scaled feature increases by one unit, the predicted value decreases by the magnitude of that coefficient. This means the feature has a negative influence on the prediction.

**Linear and Ridge MSE and R2 score comparison**
| Model | Mean Squared Error (MSE) | R² Score |
|-------|--------------------------:|---------:|
| Linear Regression | 0.3952 | 0.9709 |
| Ridge Regression | 0.3952 | 0.9709 |

**Why Ridge produce different coefficient compared to OLS**
Ridge Regression may produce a different coefficient profile than Ordinary Least Squares (OLS) Linear Regression because it applies a penalty to large coefficient values during training. This penalty shrinks the coefficients towards zero, reducing the impact of less important features and helping the model generalize better, especially when the dataset contains multicollinearity. The alpha parameter controls the strength of this penalty. A larger alpha value results in greater coefficient shrinkage, while a smaller alpha value makes Ridge Regression behave more like OLS Linear Regression.

In this dataset there is no change in the MSE and R2 score values even after setting high penalty, indicates most features are irrelevant for the data prediction. That is why the results are not changed.

## TASK 5 (a)

- check for imbalances in the data, when there is imbalance in the data, the data is getting addressed by 'class_weight = 'balanced''.

**why class_weight = 'balanced'**
- No Synthetic Data Distortion
- Prevents Data Leakage Risks
- Computational Efficiency
- Mathematically Cleaner for Logistic Regression

**Precision = TP / (TP + FP)**, where **TP** is the number of true positives and **FP** is the number of false positives. It tells how many positive predictions made are actually correct.

**Recall = TP / (TP + FN)**, where **FN** is the number of false negatives. It measures the proportion of actual positive samples that are correctly identified by the model.

For this placement classification task, **Recall** is more important because a false negative means predicting that a student will not be placed when they are actually placed. Minimizing false negatives ensures that most of the placed students are correctly identified, making the model more reliable for this task.

The **AUC (Area Under the ROC Curve)** value indicates how well the model can distinguish between the two classes. A higher AUC value means the model is better at separating placed and not placed students, while an AUC value closer to 0.5 indicates that the model performs similarly to random guessing.

## TASK 5 (b)
(a) **Precision** is calculated as **TP / (TP + FP)**, where **TP** is the number of true positives and **FP** is the number of false positives. It measures the proportion of positive predictions that are actually correct.

**Recall** is calculated as **TP / (TP + FN)**, where **FN** is the number of false negatives. It measures the proportion of actual positive samples that are correctly identified by the model.

(b) The threshold that maximizes the **F1-score** for this dataset is **0.5000**.

(c) For this placement classification task, **Recall** is more important because a false negative means predicting that a student is not placed when they are actually placed. Since missing a placed student is more costly than incorrectly predicting a student as placed, maximizing recall is the preferred objective.

(d) To optimize for recall, the classification threshold should be **lowered**. A lower threshold increases the number of positive predictions, reducing false negatives and improving recall. However, the cost of lowering the threshold is an increase in false positives, which reduces precision because more students who are not actually placed may be predicted as placed.
