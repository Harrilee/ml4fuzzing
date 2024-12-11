# %%
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
import joblib

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
    # "dateutil/test_date_parse/logs": [3],
}

model_save_dir = "models"

trainer = MutationModelTrainer(
    base_dir=base_dir,
    logs_subdirs_to_mutations=logs_subdirs_to_train,
    param_grid=param_grid,
    model_save_dir=model_save_dir,
    num_repeats=5,
    sample_size=500,
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

'''
=== Mutation 2 ===

-- Sampling Method: cluster --
Accuracies: [0.7527777777777778]
Average Accuracy: 0.7528
F1 Scores: [0.7194553019795739]
Average F1 Score: 0.7195
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7600
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.73      0.83      0.78      4732
           1       0.78      0.67      0.72      4268

    accuracy                           0.75      9000
   macro avg       0.76      0.75      0.75      9000
weighted avg       0.76      0.75      0.75      9000


-- Sampling Method: edit_distance --
Accuracies: [0.7573333333333333]
Average Accuracy: 0.7573
F1 Scores: [0.6815398075240595]
Average F1 Score: 0.6815
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7920
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.70      0.95      0.80      4732
           1       0.90      0.55      0.68      4268

    accuracy                           0.76      9000
   macro avg       0.80      0.75      0.74      9000
weighted avg       0.80      0.76      0.75      9000


-- Sampling Method: random --
Accuracies: [0.7414444444444445, 0.7507777777777778, 0.7537777777777778, 0.7587777777777778, 0.7492222222222222]
Average Accuracy: 0.7508
F1 Scores: [0.7133883483187584, 0.7054497701904137, 0.7172748150038275, 0.7152786885245902, 0.7181216435618833]
Average F1 Score: 0.7139
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 2: {'classifier__C': 1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 3: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 4: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 5: {'classifier__C': 1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7460
 Repetition 2: 0.7620
 Repetition 3: 0.7460
 Repetition 4: 0.7320
 Repetition 5: 0.7440
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.73      0.80      0.76      4732
           1       0.75      0.68      0.71      4268

    accuracy                           0.74      9000
   macro avg       0.74      0.74      0.74      9000
weighted avg       0.74      0.74      0.74      9000


--- Repetition 2 ---
              precision    recall  f1-score   support

           0       0.72      0.86      0.78      4732
           1       0.80      0.63      0.71      4268

    accuracy                           0.75      9000
   macro avg       0.76      0.74      0.74      9000
weighted avg       0.76      0.75      0.75      9000


--- Repetition 3 ---
              precision    recall  f1-score   support

           0       0.73      0.84      0.78      4732
           1       0.79      0.66      0.72      4268

    accuracy                           0.75      9000
   macro avg       0.76      0.75      0.75      9000
weighted avg       0.76      0.75      0.75      9000


--- Repetition 4 ---
              precision    recall  f1-score   support

           0       0.73      0.87      0.79      4732
           1       0.81      0.64      0.72      4268

    accuracy                           0.76      9000
   macro avg       0.77      0.75      0.75      9000
weighted avg       0.77      0.76      0.75      9000


--- Repetition 5 ---
              precision    recall  f1-score   support

           0       0.74      0.82      0.77      4732
           1       0.77      0.67      0.72      4268

    accuracy                           0.75      9000
   macro avg       0.75      0.75      0.75      9000
weighted avg       0.75      0.75      0.75      9000


-- Sampling Method: stratified --
Accuracies: [0.7662222222222222]
Average Accuracy: 0.7662
F1 Scores: [0.7224274406332454]
Average F1 Score: 0.7224
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7420
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.73      0.88      0.80      4732
           1       0.83      0.64      0.72      4268

    accuracy                           0.77      9000
   macro avg       0.78      0.76      0.76      9000
weighted avg       0.78      0.77      0.76      9000

'''