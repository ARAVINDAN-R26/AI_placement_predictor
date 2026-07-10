from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


script_folder = Path(__file__).resolve().parent
input_file = script_folder.parent / "part1" / "placementdata_prepared.csv"
output_dir = script_folder / "ml_outputs"
output_dir.mkdir(exist_ok=True)


def save_text(path, content):
    path.write_text(content, encoding="utf-8")


def print_model_metrics(metrics):
    print(metrics["model"])
    print("-" * len(metrics["model"]))
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value}")
    print()


def plot_roc_curve(y_test, probabilities, model_name, output_prefix):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_prefix}_roc_curve.png", dpi=300)
    plt.close()


def plot_residuals(y_test, predictions, probabilities, model_name, output_prefix):
    residuals = y_test.astype(int).to_numpy() - predictions.astype(int)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(probabilities, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} Residuals vs Predicted Probability")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} Residuals Distribution")
    plt.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_prefix}_residuals.png", dpi=300)
    plt.close()


def evaluate_model(model, X_test, y_test, feature_names, model_name, output_prefix):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, pos_label=1, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, pos_label=1, zero_division=0), 4),
        "f1": round(f1_score(y_test, predictions, pos_label=1, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["NotPlaced", "Placed"],
        digits=4,
    )
    save_text(output_dir / f"{output_prefix}_classification_report.txt", report)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NotPlaced", "Placed"], yticklabels=["NotPlaced", "Placed"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        feature_importance.to_csv(output_dir / f"{output_prefix}_feature_importance.csv", index=False)

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=feature_importance.head(10),
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            dodge=False,
            legend=False,
        )
        plt.title(f"{model_name} Top Feature Importances")
        plt.tight_layout()
        plt.savefig(output_dir / f"{output_prefix}_feature_importance.png", dpi=300)
        plt.close()

    return metrics


def error_analysis(y_test, predictions, probabilities, X_test_original, model_name, output_prefix):
    error_frame = X_test_original.copy()
    error_frame["actual"] = y_test.astype(int).to_numpy()
    error_frame["predicted"] = predictions.astype(int)
    error_frame["predicted_probability_placed"] = probabilities
    error_frame["correct"] = error_frame["actual"] == error_frame["predicted"]
    error_frame = error_frame[~error_frame["correct"]].copy()
    error_frame.to_csv(output_dir / f"{output_prefix}_error_analysis.csv", index=False)

    summary = (
        f"{model_name} error analysis\n"
        f"========================\n"
        f"Misclassified samples: {len(error_frame)}\n"
        f"Misclassification rate: {round(len(error_frame) / len(y_test), 4)}\n"
    )
    save_text(output_dir / f"{output_prefix}_error_summary.txt", summary)


def main():
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(input_file)

    feature_columns = [col for col in df.columns if col not in ["StudentID", "PlacementStatus"]]
    X = df[feature_columns]
    y = df["PlacementStatus"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

    # metric_choice = (
    #     "Metric choice\n"
    #     "============="
    #     "\nFor this placement prediction task, we prioritize F1-score and recall for the positive class. "
    #     "Missing a true positive (a student who should be placed) is more costly than a false alarm, "
    #     "and accuracy alone can be misleading when class balance is imperfect. ROC-AUC is also reported "
    #     "to measure how well the model ranks placed students above not-placed students."
    # )
    # save_text(output_dir / "metric_choice.txt", metric_choice)
    # print(metric_choice)

    baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    baseline_model.fit(X_train_scaled, y_train)
    baseline_metrics = evaluate_model(
        baseline_model,
        X_test_scaled,
        y_test,
        feature_columns,
        "Baseline Logistic Regression",
        "baseline",
    )
    print_model_metrics(baseline_metrics)
    plot_roc_curve(y_test, baseline_model.predict_proba(X_test_scaled)[:, 1], "Baseline Logistic Regression", "baseline")
    plot_residuals(y_test, baseline_model.predict(X_test_scaled), baseline_model.predict_proba(X_test_scaled)[:, 1], "Baseline Logistic Regression", "baseline")
    error_analysis(
        y_test,
        baseline_model.predict(X_test_scaled),
        baseline_model.predict_proba(X_test_scaled)[:, 1],
        X_test.copy(),
        "Baseline Logistic Regression",
        "baseline",
    )

    tuned_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
    )
    param_grid = {
        "n_estimators": [150, 250],
        "max_depth": [None, 6, 10],
        "min_samples_leaf": [1, 2],
    }
    grid_search = GridSearchCV(
        estimator=tuned_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    tuned_metrics = evaluate_model(
        best_model,
        X_test_scaled,
        y_test,
        feature_columns,
        f"Tuned Random Forest ({grid_search.best_params_})",
        "tuned",
    )
    print_model_metrics(tuned_metrics)
    plot_roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1], f"Tuned Random Forest", "tuned")
    plot_residuals(y_test, best_model.predict(X_test_scaled), best_model.predict_proba(X_test_scaled)[:, 1], f"Tuned Random Forest", "tuned")
    error_analysis(
        y_test,
        best_model.predict(X_test_scaled),
        best_model.predict_proba(X_test_scaled)[:, 1],
        X_test.copy(),
        f"Tuned Random Forest ({grid_search.best_params_})",
        "tuned",
    )

    comparison_df = pd.DataFrame([baseline_metrics, tuned_metrics])
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)


main()
