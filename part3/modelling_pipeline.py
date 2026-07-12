from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_part2_data():
    """Read the scaled train/test CSV files produced in Part 2."""
    project_folder = Path(__file__).resolve().parent.parent
    part2_folder = project_folder / "part2"
    X_train = pd.read_csv(part2_folder / "X_train_scaled.csv")
    X_test = pd.read_csv(part2_folder / "X_test_scaled.csv")
    y_train = pd.read_csv(part2_folder / "y_clf_train.csv").squeeze("columns")
    y_test = pd.read_csv(part2_folder / "y_clf_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def report_accuracy(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"\n{model_name}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return model, train_accuracy, test_accuracy


def main():
    X_train, X_test, y_clf_train, y_clf_test = load_part2_data()
    print("Loaded Part 2 train and test CSV files.")

    # Task 1: default Decision Tree baseline
    # ==============================================================================

    baseline_tree, baseline_train_accuracy, baseline_test_accuracy = report_accuracy("Default Decision Tree",DecisionTreeClassifier(random_state=42),X_train, y_clf_train, X_test, y_clf_test,)

   
    # Task 2: controlled Decision Tree
    # ==============================================================================

    controlled_tree, controlled_train_accuracy, controlled_test_accuracy = report_accuracy("Controlled Decision Tree (max_depth=5, min_samples_split=20)", DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42), X_train, y_clf_train, X_test, y_clf_test,)


    # Task 3: Gini versus Entropy
    # ==============================================================================
    gini_tree = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
    entropy_tree = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
    gini_tree.fit(X_train, y_clf_train)
    entropy_tree.fit(X_train, y_clf_train)
    gini_test_accuracy = accuracy_score(y_clf_test, gini_tree.predict(X_test))
    entropy_test_accuracy = accuracy_score(y_clf_test, entropy_tree.predict(X_test))
    print(f"\nGini tree test accuracy: {gini_test_accuracy:.4f}")
    print(f"Entropy tree test accuracy: {entropy_test_accuracy:.4f}")

    # Task 4: Random Forest
    # ==============================================================================
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    random_forest, rf_train_accuracy, rf_test_accuracy = report_accuracy("Random Forest", random_forest, X_train, y_clf_train, X_test, y_clf_test)
    rf_test_auc = roc_auc_score(y_clf_test, random_forest.predict_proba(X_test)[:, 1])
    feature_importance = pd.Series(random_forest.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"Random Forest test ROC-AUC: {rf_test_auc:.4f}")
    print("\nTop 5 Random Forest feature importances:")
    print(feature_importance.head(5).to_string())

    # Task 4a: Gradient Boosting
    gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_boosting, gb_train_accuracy, gb_test_accuracy = report_accuracy("Gradient Boosting", gradient_boosting, X_train, y_clf_train, X_test, y_clf_test)
    gb_test_auc = roc_auc_score(y_clf_test, gradient_boosting.predict_proba(X_test)[:, 1])
    print(f"Gradient Boosting test ROC-AUC: {gb_test_auc:.4f}")

    # Task 4b: remove the five least-important Random Forest features.
    least_important_features = feature_importance.tail(5).index.tolist()
    X_train_reduced = X_train.drop(columns=least_important_features)
    X_test_reduced = X_test.drop(columns=least_important_features)
    reduced_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    reduced_forest.fit(X_train_reduced, y_clf_train)
    reduced_rf_test_auc = roc_auc_score(y_clf_test, reduced_forest.predict_proba(X_test_reduced)[:, 1])
    print(f"\nFive removed features: {least_important_features}")
    print(f"Full Random Forest test ROC-AUC: {rf_test_auc:.4f}")
    print(f"Reduced Random Forest test ROC-AUC: {reduced_rf_test_auc:.4f}")

    # Task 5: 5-fold stratified cross-validation
    # ==============================================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_for_cv = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Controlled Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }
    cv_results = []
    for model_name, model in models_for_cv.items():
        scores = cross_val_score(model, X_train, y_clf_train, cv=cv, scoring="roc_auc")
        cv_results.append({"Model": model_name, "CV Mean AUC": scores.mean(), "CV Std AUC": scores.std()})
    cv_table = pd.DataFrame(cv_results)
    print("\n5-fold cross-validated ROC-AUC:")
    print(cv_table.to_string(index=False, float_format="{:.4f}".format))

    # Task 6: Random Forest hyperparameter tuning
    # ==============================================================================
    param_grid = {
        "randomforestclassifier__n_estimators": [50, 100, 200],
        "randomforestclassifier__max_depth": [5, 10, None],
        "randomforestclassifier__min_samples_leaf": [1, 5],
    }
    pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),RandomForestClassifier(random_state=42))
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_clf_train)
    best_pipeline = grid_search.best_estimator_
    print("\nBest Random Forest parameters:")
    print(grid_search.best_params_)
    print(f"Best cross-validated ROC-AUC: {grid_search.best_score_:.4f}")

    # Task 7: manual learning curve
    # ==============================================================================
    learning_curve_results = []
    for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
        subset_size = int(fraction * len(X_train))
        X_subset = X_train.iloc[:subset_size]
        y_subset = y_clf_train.iloc[:subset_size]
        best_pipeline.fit(X_subset, y_subset)
        train_auc = roc_auc_score(y_subset, best_pipeline.predict_proba(X_subset)[:, 1])
        test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
        learning_curve_results.append({"Training fraction": fraction, "Training AUC": train_auc, "Test AUC": test_auc})
    learning_curve_table = pd.DataFrame(learning_curve_results)
    print("\nManual learning curve:")
    print(learning_curve_table.to_string(index=False, float_format="{:.4f}".format))

    # Task 8: serialize and reload the best model.
    model_path = Path(__file__).resolve().parent / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    loaded_model = joblib.load(model_path)
    handcrafted_test_rows = pd.DataFrame(
        [dict.fromkeys(X_train.columns, 0.0), dict.fromkeys(X_train.columns, 0.5)]
    )
    reloaded_predictions = loaded_model.predict(handcrafted_test_rows)
    print(f"\nBest model saved to: {model_path}")
    print(f"Predictions for two hand-crafted rows: {reloaded_predictions}")



main()
