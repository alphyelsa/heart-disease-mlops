from datetime import datetime
import os
import shutil

import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, ConfusionMatrixDisplay, \
    PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline

from pipelines import get_preprocessing_pipeline

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(SCRIPT_DIR, "mlruns").replace("\\", "/")
TRACKING_URI = f"file:///{MLRUNS_DIR}"
mlflow.set_tracking_uri(TRACKING_URI)

ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
MODEL_DIR = os.path.join(SCRIPT_DIR, os.pardir, "model")

os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob
    }

def log_plot(path):
    mlflow.log_artifact(path)
    plt.close()


def save_confusion_matrix(model, X_test, y_test, model_name):
    cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")

    log_plot(cm_path)


def save_roc_curve(model, X_test, y_test, model_name):
    roc_path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name}.png")

    fig, ax = plt.subplots(figsize=(6,6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")

    log_plot(roc_path)

def save_pr_curve(model, X_test, y_test, model_name):
    pr_path = os.path.join(PLOTS_DIR, f"pr_curve_{model_name}.png")

    fig, ax = plt.subplots(figsize=(6,6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)

    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")

    log_plot(pr_path)



def train_and_log():

    mlflow.set_experiment("Heart_Disease_Classification")

    df = pd.read_csv("data/raw/heart.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }


    comparison_results = []

    best_auc = 0
    best_model = None
    best_name = None


    print("\n==============================")
    print("MODEL TRAINING STARTED")
    print("==============================")
    print(f"Training time: {datetime.now()}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")


    for model_name, clf in models.items():

        print("\n--------------------------------")
        print(f"Training model: {model_name}")
        print("--------------------------------")


        with mlflow.start_run(run_name=model_name):

            full_pipeline = Pipeline(
                steps=[
                    ("preprocessor", get_preprocessing_pipeline()),
                    ("classifier", clf)
                ]
            )


            full_pipeline.fit(
                X_train,
                y_train
            )


            results = evaluate_model(
                full_pipeline,
                X_test,
                y_test
            )


            # Log parameters
            mlflow.log_param(
                "model_type",
                model_name
            )


            if model_name == "RandomForest":
                mlflow.log_params({
                    "n_estimators":100,
                    "max_depth":5
                })


            if model_name == "LogisticRegression":
                mlflow.log_param(
                    "max_iter",
                    1000
                )


            # Log metrics
            mlflow.log_metrics({

                "accuracy":
                    results["accuracy"],

                "precision":
                    results["precision"],

                "recall":
                    results["recall"],

                "roc_auc":
                    results["roc_auc"]
            })


            # Save model
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name="model",
                skops_trusted_types=["numpy.dtype"]
            )


            # Store comparison result

            comparison_results.append({

                "Model":
                    model_name,

                "Accuracy":
                    round(results["accuracy"],4),

                "Precision":
                    round(results["precision"],4),

                "Recall":
                    round(results["recall"],4),

                "ROC-AUC":
                    round(results["roc_auc"],4)

            })


            print(
                f"{model_name} completed successfully"
            )

            print(
                f"ROC-AUC: {results['roc_auc']:.4f}"
            )


            if results["roc_auc"] > best_auc:

                best_auc = results["roc_auc"]
                best_model = full_pipeline
                best_name = model_name



    # Create comparison table

    comparison_df = pd.DataFrame(
        comparison_results
    )


    print("\n==============================")
    print("MODEL PERFORMANCE COMPARISON")
    print("==============================")

    print(
        comparison_df.to_string(index=False)
    )


    # Save table artifact

    comparison_path = os.path.join(
        ARTIFACT_DIR,
        "model_comparison.csv"
    )


    comparison_df.to_csv(
        comparison_path,
        index=False
    )


    mlflow.log_artifact(
        comparison_path
    )


    print("\nBest model selected:")
    print(best_name)



    # Save production model

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    mlflow.sklearn.save_model(
        sk_model=best_model,
        path="model",
        serialization_format="cloudpickle",
        skops_trusted_types=["numpy.dtype"]
    )


    print(
        "\nProduction model saved successfully"
    )
if __name__ == "__main__":
    train_and_log()