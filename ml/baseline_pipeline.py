# %%
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import warnings

# %%
# Transform the trace data into a string to be used in the CountVectorizer
class ExecTraceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.apply(lambda traces: ' '.join(traces) if isinstance(traces, list) else '')

# %%
# Define hyperparameters
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=5000)],
    },
]

# %%
# Load dataset
def get_logs(logs_dir, mutation_index):
    files = os.listdir(logs_dir)
    logs = []
    for file_name in files:
        if file_name.startswith(f"mutation{mutation_index}_"):
            with open(os.path.join(logs_dir, file_name), "r") as f:
                logs.append(json.load(f))
    # Count the number of logs
    print(f"Number of logs: {len(logs)}")
    return logs

# Combine logs into DataFrame
def combine_logs(logs):
    combined_logs = [log for log in logs if isinstance(log, dict)]
    df = pd.DataFrame(combined_logs)
    return df

# Build pipeline for model
def build_pipeline(classifier, feature_selector=SelectKBest(score_func=chi2, k=5)):
    # Pipeline for preprocessing text in 'exec_trace' column
    exec_trace_pipeline = Pipeline([
        ('exec_transform', ExecTraceTransformer()),
        ('vectorizer', CountVectorizer())
    ])

    preprocessor = ColumnTransformer([
        ('exec_trace', exec_trace_pipeline, 'exec_trace'),
    ], remainder='drop')

    # Pipeline for each model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', feature_selector),
        ('classifier', classifier)
    ])

    return pipeline

# Train model, hyperparameter tuning with GridSearchCV
def train_model(X_train, y_train, pipeline, param_grid):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    return grid_search

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report
    }

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

# %%
class MutationModelTrainer:
    def __init__(self, base_dir, logs_subdirs_to_mutations, param_grid, model_save_dir="models",
                 num_repeats=5, sample_size=2000, seed_start=42):
        self.base_dir = base_dir
        self.logs_subdirs_to_mutations = logs_subdirs_to_mutations
        self.param_grid = param_grid
        self.model_save_dir = model_save_dir
        self.num_repeats = num_repeats
        self.sample_size = sample_size
        self.seed_start = seed_start
        self.results = {}

        os.makedirs(self.model_save_dir, exist_ok=True)

    def process_mutation(self, logs_subdir, mutation_index):
        logs_dir = os.path.join(self.base_dir, logs_subdir)
        print(f"\nProcessing Logs Subdir: '{logs_subdir}', Mutation Index: {mutation_index}")

        project_name = logs_subdir.split('/')[-3]
        print(f"Project Name: {project_name}")

        logs = get_logs(logs_dir, mutation_index)

        if not logs:
            print(f"No logs found for mutation index {mutation_index} in '{logs_subdir}'. Skipping.")
            return

        df = combine_logs(logs)

        df = df.dropna(subset=['exec_trace', 'verdict'])

        # Encode the verdict: 'pass' as 1, others as 0
        y = df['verdict'].apply(lambda x: 1 if x.lower() == 'pass' else 0)

        X = df[['exec_trace']]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        accuracies = []
        f1_scores = []
        classification_reports = []
        random_seeds = []

        for i in range(self.num_repeats):
            current_seed = self.seed_start + i
            random_seeds.append(current_seed)
            print(f"\n--- Repetition {i+1}/{self.num_repeats} ---")

            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=self.sample_size,
                random_state=current_seed,
                stratify=y_train_full
            )

            # Print train size and test size
            print(f"Train Size: {len(X_train)}")
            print(f"Test Size: {len(X_test)}")

            # Build the pipeline
            placeholder_classifier = LogisticRegression()
            pipeline = build_pipeline(classifier=placeholder_classifier)

            # Train the model
            grid_search = train_model(X_train, y_train, pipeline, self.param_grid)

            # Evaluate the best model
            evaluation = evaluate_model(grid_search.best_estimator_, X_test, y_test)

            # Save the best model for this repetition
            model_filename = os.path.join(
                self.model_save_dir,
                f"{project_name}_mutation_{mutation_index}_repeat_{i+1}.pkl"
            )
            save_model(grid_search.best_estimator_, model_filename)

            # Collect metrics
            accuracies.append(evaluation['accuracy'])
            f1_scores.append(evaluation['f1_score'])
            classification_reports.append(evaluation['classification_report'])

            print("Best Parameters:")
            print(grid_search.best_params_)
            print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
            print(f"Test Set Accuracy: {evaluation['accuracy']:.4f}")
            print(f"Test Set F1 Score: {evaluation['f1_score']:.4f}")
            print("Classification Report:")
            print(evaluation['classification_report'])

        # Average results
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)

        self.results[mutation_index] = {
            'random_seeds': random_seeds,
            'accuracies': accuracies,
            'average_accuracy': avg_accuracy,
            'f1_scores': f1_scores,
            'average_f1_score': avg_f1,
            'classification_reports': classification_reports
        }

        print(f"\n=== Final Results for Mutation {mutation_index} ===")
        print(f"Random Seeds Used: {random_seeds}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")

    def train_all(self):
        for logs_subdir, mutation_indices in self.logs_subdirs_to_mutations.items():
            for mutation_index in mutation_indices:
                self.process_mutation(logs_subdir, mutation_index)

    def get_results(self):
        return self.results
# %%
base_dir = "../fuzz_test"

# Subdirectory path: [mutation indices]
logs_subdirs_to_train = {
    "textdistance/test_DamerauLevenshtein/logs": [2],
    "dateutil/test_date_parse/logs": [3],
}

model_save_dir = "models"

trainer = MutationModelTrainer(
    base_dir=base_dir,
    logs_subdirs_to_mutations=logs_subdirs_to_train,
    param_grid=param_grid,
    model_save_dir=model_save_dir,
    num_repeats=5,
    sample_size=100,
    seed_start=42
)

trainer.train_all()

results = trainer.get_results()

# %%
for mutation, res in results.items():
    print(f"\nMutation {mutation}:")
    print(f"Random Seeds Used: {res['random_seeds']}")
    print(f"Accuracies: {res['accuracies']}")
    print(f"Average Accuracy: {res['average_accuracy']:.4f}")
    print(f"F1 Scores: {res['f1_scores']}")
    print(f"Average F1 Score: {res['average_f1_score']:.4f}")
    print("Classification Reports:")
    for idx, report in enumerate(res['classification_reports'], 1):
        print(f"\n--- Classification Report for Repetition {idx} ---")
        print(report)
