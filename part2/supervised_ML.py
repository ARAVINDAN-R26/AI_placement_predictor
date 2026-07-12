import os
import matplotlib.pyplot as plt
import numpy as np
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
   
    script_folder = Path(__file__).resolve().parent

    # Go to the folder before this one
    os.chdir(script_folder.parent)

    # Go into the part1 folder
    os.chdir("part1")

    # Read the CSV file
    data = pd.read_csv("cleaned_data.csv")
    return data


def make_labels(data):

    # Regression target: a number
    y_reg = data["CGPA"].astype(float)

    # Classification target: use the existing binary column
    y_clf = data["PlacementStatus"].astype(str).str.lower()
    y_clf = (y_clf == "placed").astype(int)

    # Features: everything except the two target columns
    X = data.drop(columns=["CGPA", "PlacementStatus"])

    return X, y_reg, y_clf


def encode_categorical_features(X):
    
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        return X

    # These columns are yes/no indicators, so they do not have a natural order.
    # One-hot encoding is used to avoid pretending that one category is "higher" than the other.
    encoded_X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return encoded_X


def bootstrap_auc_difference(y_test, baseline_probabilities, regularized_probabilities):
    
    auc_differences = []

    for i in range(500):
        sample_indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
        sampled_y = np.asarray(y_test)[sample_indices]
        baseline_auc = roc_auc_score(sampled_y, baseline_probabilities[sample_indices])
        regularized_auc = roc_auc_score(sampled_y, regularized_probabilities[sample_indices])
        auc_differences.append(baseline_auc - regularized_auc)

    mean_difference = np.mean(auc_differences)
    lower_bound, upper_bound = np.percentile(auc_differences, [2.5, 97.5])

    print("\nBootstrap AUC difference: C=1.0 minus C=0.01")
    print(f"Mean AUC difference: {mean_difference:.4f}")
    print(f"95% confidence interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

    return mean_difference, lower_bound, upper_bound


def regularization_experiment(X_train, y_train, X_test, y_test, class_weight=None):
    
    models = [
        ("Baseline (C=1.0)", LogisticRegression(C=1.0, max_iter=1000, class_weight=class_weight)),
        ("Strong L2 regularization (C=0.01)", LogisticRegression(C=0.01, max_iter=1000, class_weight=class_weight)),
    ]
    results = []
    probabilities_by_model = {}

    for model_name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        probabilities_by_model[model_name] = probabilities
        results.append({
            "Model": model_name,
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, probabilities),
        })

    comparison_table = pd.DataFrame(results)
    print("\nTask 6: Logistic Regression regularization comparison")
    print(comparison_table.to_string(index=False, float_format="{:.4f}".format))

    return (
        comparison_table,
        probabilities_by_model["Baseline (C=1.0)"],
        probabilities_by_model["Strong L2 regularization (C=0.01)"],
    )


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
    print(class_counts)
    minority_percentage = class_counts.min() / class_counts.sum()
    print("\nClassification training-label counts:")
    

    if minority_percentage < 0.35:
        class_weight = "balanced"
        logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight)
        print("Using class_weight='balanced' because the minority class has fewer than 35% of samples.")
    else:
        class_weight = None
        logistic_model = LogisticRegression(max_iter=1000, class_weight=class_weight)
        print("No class weighting is needed because both classes have at least 35% of samples.")

    logistic_model.fit(X_train_scaled, y_clf_train)
    y_pred_clf = logistic_model.predict(X_test_scaled)

    #getting the probablity for every value and slicing only placed column
    y_prob_clf = logistic_model.predict_proba(X_test_scaled)[:, 1] 

    print("\nConfusion matrix:")
    print(confusion_matrix(y_clf_test, y_pred_clf))
    print("\nClassification report:")
    print(classification_report(y_clf_test, y_pred_clf, digits=4))

    fpr, tpr, threshold = roc_curve(y_clf_test, y_prob_clf)
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
    roc_plot_path = Path(__file__).resolve().parent / "logistic_regression_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    print(f"ROC curve saved to: {roc_plot_path}")
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

    #TASK 6
    #=======================================================================
    regularization_comparison, baseline_probabilities, regularized_probabilities = regularization_experiment(X_train_scaled, y_clf_train, X_test_scaled, y_clf_test, class_weight)
    print("regularization experiment completed")

    #TASK 7
    #=======================================================================
    mean_auc_difference, lower_ci, upper_ci = bootstrap_auc_difference(y_clf_test, baseline_probabilities, regularized_probabilities)

    print(f"Mean accuracy difference: {mean_auc_difference:.2f}")
    print(f"Lower bound: {lower_ci:.2f}")
    print(f"Upper bound: {upper_ci:.2f}")

main()
