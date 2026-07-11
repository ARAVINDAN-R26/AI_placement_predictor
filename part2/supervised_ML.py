import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cleaned_data():
    """Move to the parent folder, then open the part1 folder, and read the CSV file."""
    script_folder = Path(__file__).resolve().parent

    # Go to the folder before this one
    os.chdir(script_folder.parent)

    # Go into the part1 folder
    os.chdir("part1")

    # Read the CSV file
    data = pd.read_csv("cleaned_data.csv")
    return data


def make_labels(data):
    """Take one regression column and one classification column."""

    # Regression target: a number
    y_reg = data["CGPA"].astype(float)

    # Classification target: use the existing binary column
    y_clf = data["PlacementStatus"].astype(str).str.lower()
    y_clf = (y_clf == "placed").astype(int)

    # Features: everything except the two target columns
    X = data.drop(columns=["CGPA", "PlacementStatus"])

    return X, y_reg, y_clf


def encode_categorical_features(X):
    """Encode categorical columns using one-hot encoding for non-ordered categories."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        return X

    # These columns are yes/no indicators, so they do not have a natural order.
    # One-hot encoding is used to avoid pretending that one category is "higher" than the other.
    encoded_X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return encoded_X


def main():

    #TASK 1
    #=======================================================================
    data = load_cleaned_data()
    X, y_reg, y_clf = make_labels(data)
    print("Completed Loading and categorizing data")

    #TASK 2
    #=======================================================================
    X = encode_categorical_features(X)
    print("Successfully encoded the categorical values")

    #TASK 3
    #=======================================================================
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("The datas are successfully scaled")

    #TASK 4
    #=======================================================================
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_reg_train)
    y_pred_reg = linear_model.predict(X_test_scaled)

    linear_mse = mean_squared_error(y_reg_test, y_pred_reg)
    linear_r2 = r2_score(y_reg_test, y_pred_reg)
    print(f"Linear Regression MSE: {linear_mse:.4f}")
    print(f"Linear Regression R2: {linear_r2:.4f}")

    coefficients = pd.Series(linear_model.coef_, index=X.columns)
    print("\nLinear Regression coefficients:")
    print(coefficients.to_string())

    top_three = coefficients.abs().nlargest(3).index
    print("\nThree features with the largest absolute coefficients:")
    for feature in top_three:
        print(f"{feature}: {coefficients[feature]:.4f}")

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_reg_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    ridge_mse = mean_squared_error(y_reg_test, y_pred_ridge)
    ridge_r2 = r2_score(y_reg_test, y_pred_ridge)
    print(f"\nRidge Regression MSE: {ridge_mse:.4f}")
    print(f"Ridge Regression R2: {ridge_r2:.4f}")

    #TASK 5
    #=======================================================================

    #PART A

    # Logistic Regression classification
    class_counts = y_clf_train.value_counts()
    minority_percentage = class_counts.min() / class_counts.sum()
    print("\nClassification training-label counts:")
    print(class_counts)

    if minority_percentage < 0.35:
        logistic_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        print("Using class_weight='balanced' because the minority class has fewer than 35% of samples.")
    else:
        logistic_model = LogisticRegression(max_iter=1000)
        print("No class weighting is needed because both classes have at least 35% of samples.")

    logistic_model.fit(X_train_scaled, y_clf_train)
    y_pred_clf = logistic_model.predict(X_test_scaled)

    #getting the probablity for roc auc curve
    y_prob_clf = logistic_model.predict_proba(X_test_scaled)[:, 1] 

    print("\nConfusion matrix:")
    print(confusion_matrix(y_clf_test, y_pred_clf))
    print("\nClassification report:")
    print(classification_report(y_clf_test, y_pred_clf, digits=4))

    fpr, tpr, _ = roc_curve(y_clf_test, y_prob_clf)
    auc_score = roc_auc_score(y_clf_test, y_prob_clf)
    print(f"ROC-AUC: {auc_score:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="Logistic Regression ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curve")
    plt.text(0.60, 0.15, f"AUC = {auc_score:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PART B
    threshold_results = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        threshold_predictions = (y_prob_clf >= threshold).astype(int)
        threshold_results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(y_clf_test, threshold_predictions),
                "Recall": recall_score(y_clf_test, threshold_predictions),
                "F1": f1_score(y_clf_test, threshold_predictions),
            }
        )

    threshold_table = pd.DataFrame(threshold_results)
    print("\nDecision-threshold sensitivity:")
    print(threshold_table.to_string(index=False, float_format="{:.4f}".format))

main()
