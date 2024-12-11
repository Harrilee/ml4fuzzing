import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
from enum import Enum
from textdistance import DamerauLevenshtein, Jaccard, Cosine

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

class DataSelectionMethod(Enum):
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    CLUSTER = 'cluster'
    EDIT_DISTANCE = 'edit_distance'
    JACCARD_DISTANCE = 'jaccard_distance'
    EMBEDDING_EUCLIDEAN_DISTANCE = 'embedding_euclidean_distance'
    COSINE_SIMILARITY = 'cosine_similarity'

# Custom transformer to convert exec_trace lists to strings
class ExecTraceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X['exec_trace']  # Explicitly select 'exec_trace'
        return X.apply(lambda traces: ' '.join(traces) if isinstance(traces, list) else '')

# Function to load logs
def get_logs(logs_dir, mutation_index):
    files = os.listdir(logs_dir)
    logs = []
    for file_name in files:
        if file_name.startswith(f"mutation{mutation_index}_") and file_name.endswith('.json'):
            with open(os.path.join(logs_dir, file_name), "r", encoding='utf-8') as f:
                try:
                    logs.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"Warning: Unable to decode file {file_name}. Skipping.")
    print(f"Loaded {len(logs)} log files.")
    return logs

# Combine log data into a DataFrame
def combine_logs(logs):
    combined_logs = [log for log in logs if isinstance(log, dict)]
    df = pd.DataFrame(combined_logs)
    return df

# Define data selection methods
def random_sampling(X, y, sample_size, random_state=42):
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=sample_size,
        random_state=random_state,
        stratify=None
    )
    return X_sample, y_sample

def stratified_sampling(X, y, sample_size, random_state=42):
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=sample_size,
        random_state=random_state,
        stratify=y
    )
    return X_sample, y_sample


def cluster_based_sampling(X, y, sample_size, n_clusters=10, random_state=42):
    # Apply ExecTraceTransformer to convert exec_trace to string
    transformer = ExecTraceTransformer()
    X_transformed = transformer.transform(X)

    vectorizer = CountVectorizer()
    X_vectors = vectorizer.fit_transform(X_transformed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_vectors)

    df = X.copy()
    df['cluster'] = clusters
    df['y'] = y.values

    samples_per_cluster = max(1, sample_size // n_clusters)

    # Perform sampling without resetting the index
    sampled_df = df.groupby('cluster').apply(
        lambda x: x.sample(n=min(len(x), samples_per_cluster), random_state=random_state)
    )

    # sampled_df now has a MultiIndex (cluster, original_index)
    # To get the original indices, use `level=1`
    sampled_indices = sampled_df.index.get_level_values(1)

    # Drop the sampled indices from df to get the remaining DataFrame
    remaining_df = df.drop(sampled_indices)

    # Calculate how many additional samples are needed
    current_sample_size = len(sampled_df)
    if current_sample_size < sample_size:
        additional_samples = sample_size - current_sample_size
        additional_df = remaining_df.sample(n=additional_samples, random_state=random_state)
        sampled_df = pd.concat([sampled_df, additional_df], ignore_index=False)

    # Return the sampled exec_trace and corresponding y
    return sampled_df[['exec_trace']], sampled_df['y']

def edit_distance_sampling(X, y, sample_size, random_state=42):
    X_inputs = X['input']

    selected_indices = []
    remaining_indices = list(range(len(X_inputs)))

    # Select the first random sample
    np.random.seed(random_state)
    first_index = np.random.choice(remaining_indices)
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)

    # Precompute all pairwise distances using 'input'
    distance_matrix = np.zeros((len(X_inputs), len(X_inputs)))
    for i in range(len(X_inputs)):
        for j in range(i + 1, len(X_inputs)):
            print(f"\rPrecomputing edit distance: {i+1}/{len(X_inputs)}", end='')
            distance = DamerauLevenshtein().distance(str(X_inputs.iloc[i]), str(X_inputs.iloc[j]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    print("\nPrecomputed all pairwise edit distances based on 'input'.")

    # Iteratively select samples farthest from the already selected set
    while len(selected_indices) < sample_size and remaining_indices:
        # Calculate distances to the selected set
        distances = np.min(distance_matrix[remaining_indices][:, selected_indices], axis=1)
        farthest_idx_relative = np.argmax(distances)
        farthest_index = remaining_indices[farthest_idx_relative]
        selected_indices.append(farthest_index)
        remaining_indices.remove(farthest_index)

    sampled_df = X.iloc[selected_indices].copy()
    sampled_y = y.iloc[selected_indices].copy()

    return sampled_df[['exec_trace']], sampled_y

def jaccard_distance_sampling(X, y, sample_size, random_state=42):
    X_input = X['input']

    token_sets = X_input.apply(lambda trace: set(' '.join(trace).split()) if isinstance(trace, list) else set(trace.split()))

    selected_indices = []
    remaining_indices = list(range(len(token_sets)))

    # Select the first random sample
    np.random.seed(random_state)
    first_index = np.random.choice(remaining_indices)
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)

    # Precompute all pairwise Jaccard distances
    distance_matrix = np.zeros((len(token_sets), len(token_sets)))
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            print(f"\rPrecomputing Jaccard distance: {i+1}/{len(token_sets)}", end='')
            intersection = token_sets.iloc[i].intersection(token_sets.iloc[j])
            union = token_sets.iloc[i].union(token_sets.iloc[j])
            distance = 1 - (len(intersection) / len(union)) if union else 0
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    print("\nPrecomputed all pairwise Jaccard distances based on 'input'.")

    while len(selected_indices) < sample_size and remaining_indices:
        distances = np.min(distance_matrix[remaining_indices][:, selected_indices], axis=1)
        farthest_idx_relative = np.argmax(distances)
        farthest_index = remaining_indices[farthest_idx_relative]
        selected_indices.append(farthest_index)
        remaining_indices.remove(farthest_index)

    sampled_df = X.iloc[selected_indices].copy()
    sampled_y = y.iloc[selected_indices].copy()

    return sampled_df[['exec_trace']], sampled_y

def embedding_euclidean_distance_sampling(X, y, sample_size, random_state=42):
    X_input = X['input'].apply(lambda traces: ' '.join(traces) if isinstance(traces, list) else '')

    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(X_input)

    X_dense = X_vectors.toarray()

    selected_indices = []
    remaining_indices = list(range(len(X_dense)))

    np.random.seed(random_state)
    first_index = np.random.choice(remaining_indices)
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)

    print("Computing Euclidean distance matrix...")
    distance_matrix = euclidean_distances(X_dense)

    print("Precomputed all pairwise Euclidean distances based on 'input'.")

    while len(selected_indices) < sample_size and remaining_indices:
        distances = np.min(distance_matrix[remaining_indices][:, selected_indices], axis=1)
        farthest_idx_relative = np.argmax(distances)
        farthest_index = remaining_indices[farthest_idx_relative]
        selected_indices.append(farthest_index)
        remaining_indices.remove(farthest_index)

    sampled_df = X.iloc[selected_indices].copy()
    sampled_y = y.iloc[selected_indices].copy()

    return sampled_df[['exec_trace']], sampled_y

def cosine_similarity_sampling(X, y, sample_size, random_state=42):
    X_input = X['input'].apply(lambda traces: ' '.join(traces) if isinstance(traces, list) else '')

    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(X_input)

    X_dense = X_vectors.toarray()

    selected_indices = []
    remaining_indices = list(range(len(X_dense)))

    np.random.seed(random_state)
    first_index = np.random.choice(remaining_indices)
    selected_indices.append(first_index)
    remaining_indices.remove(first_index)

    print("Computing cosine similarity matrix...")
    similarity_matrix = cosine_similarity(X_dense)

    distance_matrix = 1 - similarity_matrix

    print("Precomputed all pairwise cosine distances based on 'input'.")

    while len(selected_indices) < sample_size and remaining_indices:
        distances = np.min(distance_matrix[remaining_indices][:, selected_indices], axis=1)
        farthest_idx_relative = np.argmax(distances)
        farthest_index = remaining_indices[farthest_idx_relative]
        selected_indices.append(farthest_index)
        remaining_indices.remove(farthest_index)

    sampled_df = X.iloc[selected_indices].copy()
    sampled_y = y.iloc[selected_indices].copy()

    return sampled_df[['exec_trace']], sampled_y

# Main training class
class MutationModelTrainer:
    def __init__(self, base_dir, logs_subdirs_to_mutations, param_grid, model_save_dir="models",
                 num_repeats=5, sample_size=500, seed_start=42):
        self.base_dir = base_dir
        self.logs_subdirs_to_mutations = logs_subdirs_to_mutations
        self.param_grid = param_grid
        self.model_save_dir = model_save_dir
        self.num_repeats = num_repeats
        self.sample_size = sample_size
        self.seed_start = seed_start
        self.results = {}

        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_and_prepare_data(self, logs_subdir, mutation_index):
        logs_dir = os.path.join(self.base_dir, logs_subdir)
        print(f"\nProcessing Logs Subdir: '{logs_subdir}', Mutation Index: {mutation_index}")

        # Extract project name, assuming it's part of the directory structure
        project_name = logs_subdir.split(os.sep)[-3] if len(logs_subdir.split(os.sep)) >= 3 else logs_subdir.split('/')[-3]
        print(f"Project Name: {project_name}")

        logs = get_logs(logs_dir, mutation_index)

        if not logs:
            print(f"No logs found for mutation index {mutation_index} in '{logs_subdir}'. Skipping.")
            return None, None, None, None, project_name

        df = combine_logs(logs)
        df = df.dropna(subset=['exec_trace', 'verdict', 'input'])

        # Encode verdict: 'pass' as 1, others as 0
        y = df['verdict'].apply(lambda x: 1 if x.lower() == 'pass' else 0)
        X = df[['exec_trace', 'input']]  # Include both 'exec_trace' and 'input'

        # Initial train-test split
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.9, random_state=42, stratify=y
        )

        return X_train_full, X_test, y_train_full, y_test, project_name

    def select_data(self, X, y, method, random_state):
        if method == DataSelectionMethod.RANDOM.value:
            return random_sampling(X[['exec_trace']], y, self.sample_size, random_state)
        elif method == DataSelectionMethod.STRATIFIED.value:
            return stratified_sampling(X[['exec_trace']], y, self.sample_size, random_state)
        elif method == DataSelectionMethod.CLUSTER.value:
            return cluster_based_sampling(X[['exec_trace']], y, self.sample_size, n_clusters=10, random_state=random_state)
        elif method == DataSelectionMethod.EDIT_DISTANCE.value:
            return edit_distance_sampling(X, y, self.sample_size, random_state)
        elif method == DataSelectionMethod.JACCARD_DISTANCE.value:
            return jaccard_distance_sampling(X, y, self.sample_size, random_state)
        elif method == DataSelectionMethod.COSINE_SIMILARITY.value:
            return cosine_similarity_sampling(X, y, self.sample_size, random_state)
        elif method == DataSelectionMethod.EMBEDDING_EUCLIDEAN_DISTANCE.value:
            return embedding_euclidean_distance_sampling(X, y, self.sample_size, random_state)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, random_state):
        # Build pipeline
        pipeline = Pipeline([
            ('exec_transform', ExecTraceTransformer()),
            ('vectorizer', CountVectorizer()),
            ('feature_selection', SelectKBest(score_func=chi2, k=5)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
        ])

        # Define GridSearch parameters
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.param_grid,
            cv=2,  # Set to 2-fold cross-validation for faster training
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        # Train the model
        grid_search.fit(X_train, y_train)

        # Predict
        y_pred = grid_search.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, f1, report, grid_search.best_params_, grid_search.best_score_

    def process_mutation(self, logs_subdir, mutation_index, methods):
        X_train_full, X_test, y_train_full, y_test, project_name = self.load_and_prepare_data(logs_subdir, mutation_index)
        if X_train_full is None:
            return

        for method in methods:
            print(f"\n=== Using Data Selection Method: {method} ===")
            accuracies = []
            f1_scores = []
            classification_reports = []
            best_params = []
            best_cv_scores = []

            repetitions = self.num_repeats if method in [DataSelectionMethod.RANDOM.value,
                                                       DataSelectionMethod.CLUSTER.value,
                                                       DataSelectionMethod.STRATIFIED.value] else 1

            for i in range(1, repetitions + 1):
                random_state = self.seed_start + i
                print(f"\n--- Repetition {i}/{repetitions} ---")

                X_sample, y_sample = self.select_data(X_train_full, y_train_full, method, random_state)

                print(f"Training Set Size: {len(X_sample)}")
                print(f"Test Set Size: {len(X_test)}")

                accuracy, f1, report, params, cv_score = self.train_and_evaluate(
                    X_sample, y_sample, X_test, y_test,
                    random_state
                )

                accuracies.append(accuracy)
                f1_scores.append(f1)
                classification_reports.append(report)
                best_params.append(params)
                best_cv_scores.append(cv_score)

                # Print report
                print(f"Accuracy: {accuracy:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print("Classification Report:")
                print(report)

            # Store results
            if mutation_index not in self.results:
                self.results[mutation_index] = {}
            self.results[mutation_index][method] = {
                'accuracies': accuracies,
                'average_accuracy': np.mean(accuracies),
                'f1_scores': f1_scores,
                'average_f1_score': np.mean(f1_scores),
                'classification_reports': classification_reports,
                'best_params': best_params,
                'best_cv_scores': best_cv_scores
            }

    def train_all(self):
        # Define data selection methods to compare
        methods = [
            DataSelectionMethod.RANDOM.value,
            DataSelectionMethod.STRATIFIED.value,
            DataSelectionMethod.CLUSTER.value,
            DataSelectionMethod.EDIT_DISTANCE.value,
            DataSelectionMethod.COSINE_SIMILARITY.value,
            DataSelectionMethod.JACCARD_DISTANCE.value,
            DataSelectionMethod.EMBEDDING_EUCLIDEAN_DISTANCE.value,
        ]

        for logs_subdir, mutation_indices in self.logs_subdirs_to_mutations.items():
            for mutation_index in mutation_indices:
                self.process_mutation(logs_subdir, mutation_index, methods)

    def get_results(self):
        return self.results

# Define GridSearch parameters
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs']
}

# Set base directory and log subdirectories with mutation indices
base_dir = "../fuzz_test"

# Subdirectory paths: [Mutation Indices]
logs_subdirs_to_train = {
    "textdistance/test_DamerauLevenshtein/logs": [2],
    # "dateutil/test_date_parse/logs": [3],  # Uncomment if needed
}

model_save_dir = "models"

# Instantiate trainer
trainer = MutationModelTrainer(
    base_dir=base_dir,
    logs_subdirs_to_mutations=logs_subdirs_to_train,
    param_grid=param_grid,
    model_save_dir=model_save_dir,
    num_repeats=5,
    sample_size=100,
    seed_start=42
)

# Train all models
trainer.train_all()

# Retrieve and display results
results = trainer.get_results()

for mutation, methods in results.items():
    print(f"\n=== Mutation {mutation} ===")
    for method, metrics in methods.items():
        print(f"\n-- Sampling Method: {method} --")
        print(f"Accuracies: {metrics['accuracies']}")
        print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
        print(f"F1 Scores: {metrics['f1_scores']}")
        print(f"Average F1 Score: {metrics['average_f1_score']:.4f}")
        print("Best Parameters per Repetition:")
        for i, params in enumerate(metrics['best_params'], 1):
            print(f" Repetition {i}: {params}")
        print("Best CV Scores per Repetition:")
        for i, cv_score in enumerate(metrics['best_cv_scores'], 1):
            print(f" Repetition {i}: {cv_score:.4f}")
        print("Classification Reports:")
        for i, report in enumerate(metrics['classification_reports'], 1):
            print(f"\n--- Repetition {i} ---")
            print(report)

'''
=== Mutation 2 ===

-- Sampling Method: random --
Accuracies: [0.6847777777777778, 0.7575555555555555, 0.7536666666666667, 0.7501111111111111, 0.7633333333333333]
Average Accuracy: 0.7419
F1 Scores: [0.7161580790395198, 0.6857718894009217, 0.7109140696309819, 0.6846164633291264, 0.7264320575391728]
Average F1 Score: 0.7048
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 2: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 3: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 4: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 5: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7100
 Repetition 2: 0.7800
 Repetition 3: 0.6800
 Repetition 4: 0.7000
 Repetition 5: 0.7300
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.79      0.55      0.65      4732
           1       0.62      0.84      0.72      4268

    accuracy                           0.68      9000
   macro avg       0.71      0.69      0.68      9000
weighted avg       0.71      0.68      0.68      9000


--- Repetition 2 ---
              precision    recall  f1-score   support

           0       0.70      0.94      0.80      4732
           1       0.89      0.56      0.69      4268

    accuracy                           0.76      9000
   macro avg       0.80      0.75      0.74      9000
weighted avg       0.79      0.76      0.75      9000


--- Repetition 3 ---
              precision    recall  f1-score   support

           0       0.72      0.86      0.79      4732
           1       0.80      0.64      0.71      4268

    accuracy                           0.75      9000
   macro avg       0.76      0.75      0.75      9000
weighted avg       0.76      0.75      0.75      9000


--- Repetition 4 ---
              precision    recall  f1-score   support

           0       0.70      0.91      0.79      4732
           1       0.85      0.57      0.68      4268

    accuracy                           0.75      9000
   macro avg       0.78      0.74      0.74      9000
weighted avg       0.77      0.75      0.74      9000


--- Repetition 5 ---
              precision    recall  f1-score   support

           0       0.74      0.85      0.79      4732
           1       0.80      0.66      0.73      4268

    accuracy                           0.76      9000
   macro avg       0.77      0.76      0.76      9000
weighted avg       0.77      0.76      0.76      9000


-- Sampling Method: stratified --
Accuracies: [0.7595555555555555, 0.7504444444444445, 0.7685555555555555, 0.7557777777777778, 0.7451111111111111]
Average Accuracy: 0.7559
F1 Scores: [0.7219938335046249, 0.6868377021751255, 0.7273917026567204, 0.7230342741935484, 0.7153139736907421]
Average F1 Score: 0.7149
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 2: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 3: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 4: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 5: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7200
 Repetition 2: 0.7700
 Repetition 3: 0.7400
 Repetition 4: 0.7400
 Repetition 5: 0.7800
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.73      0.85      0.79      4732
           1       0.80      0.66      0.72      4268

    accuracy                           0.76      9000
   macro avg       0.77      0.75      0.76      9000
weighted avg       0.76      0.76      0.76      9000


--- Repetition 2 ---
              precision    recall  f1-score   support

           0       0.70      0.91      0.79      4732
           1       0.85      0.58      0.69      4268

    accuracy                           0.75      9000
   macro avg       0.78      0.74      0.74      9000
weighted avg       0.77      0.75      0.74      9000


--- Repetition 3 ---
              precision    recall  f1-score   support

           0       0.74      0.87      0.80      4732
           1       0.82      0.65      0.73      4268

    accuracy                           0.77      9000
   macro avg       0.78      0.76      0.76      9000
weighted avg       0.78      0.77      0.76      9000


--- Repetition 4 ---
              precision    recall  f1-score   support

           0       0.74      0.83      0.78      4732
           1       0.78      0.67      0.72      4268

    accuracy                           0.76      9000
   macro avg       0.76      0.75      0.75      9000
weighted avg       0.76      0.76      0.75      9000


--- Repetition 5 ---
              precision    recall  f1-score   support

           0       0.73      0.81      0.77      4732
           1       0.76      0.68      0.72      4268

    accuracy                           0.75      9000
   macro avg       0.75      0.74      0.74      9000
weighted avg       0.75      0.75      0.74      9000


-- Sampling Method: cluster --
Accuracies: [0.6815555555555556, 0.6657777777777778, 0.7476666666666667, 0.758, 0.7668888888888888]
Average Accuracy: 0.7240
F1 Scores: [0.6701197053406999, 0.6732565718009993, 0.7183430484931167, 0.7289021657953697, 0.7264667535853977]
Average F1 Score: 0.7034
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 2: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 3: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 4: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
 Repetition 5: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.6700
 Repetition 2: 0.7200
 Repetition 3: 0.8100
 Repetition 4: 0.6900
 Repetition 5: 0.6800
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.70      0.68      0.69      4732
           1       0.66      0.68      0.67      4268

    accuracy                           0.68      9000
   macro avg       0.68      0.68      0.68      9000
weighted avg       0.68      0.68      0.68      9000


--- Repetition 2 ---
              precision    recall  f1-score   support

           0       0.71      0.61      0.66      4732
           1       0.63      0.73      0.67      4268

    accuracy                           0.67      9000
   macro avg       0.67      0.67      0.67      9000
weighted avg       0.67      0.67      0.67      9000


--- Repetition 3 ---
              precision    recall  f1-score   support

           0       0.74      0.81      0.77      4732
           1       0.76      0.68      0.72      4268

    accuracy                           0.75      9000
   macro avg       0.75      0.74      0.74      9000
weighted avg       0.75      0.75      0.75      9000


--- Repetition 4 ---
              precision    recall  f1-score   support

           0       0.74      0.82      0.78      4732
           1       0.78      0.69      0.73      4268

    accuracy                           0.76      9000
   macro avg       0.76      0.75      0.76      9000
weighted avg       0.76      0.76      0.76      9000


--- Repetition 5 ---
              precision    recall  f1-score   support

           0       0.74      0.87      0.80      4732
           1       0.82      0.65      0.73      4268

    accuracy                           0.77      9000
   macro avg       0.78      0.76      0.76      9000
weighted avg       0.77      0.77      0.76      9000


-- Sampling Method: edit_distance --
Accuracies: [0.5657777777777778]
Average Accuracy: 0.5658
F1 Scores: [0.5163366336633664]
Average F1 Score: 0.5163
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.8000
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.58      0.64      0.61      4732
           1       0.55      0.49      0.52      4268

    accuracy                           0.57      9000
   macro avg       0.56      0.56      0.56      9000
weighted avg       0.56      0.57      0.56      9000


-- Sampling Method: cosine_similarity --
Accuracies: [0.647]
Average Accuracy: 0.6470
F1 Scores: [0.6677820767541567]
Average F1 Score: 0.6678
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7200
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.71      0.56      0.62      4732
           1       0.60      0.75      0.67      4268

    accuracy                           0.65      9000
   macro avg       0.66      0.65      0.65      9000
weighted avg       0.66      0.65      0.64      9000


-- Sampling Method: jaccard_distance --
Accuracies: [0.6336666666666667]
Average Accuracy: 0.6337
F1 Scores: [0.6458266194005801]
Average F1 Score: 0.6458
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.6900
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.68      0.57      0.62      4732
           1       0.60      0.70      0.65      4268

    accuracy                           0.63      9000
   macro avg       0.64      0.64      0.63      9000
weighted avg       0.64      0.63      0.63      9000


-- Sampling Method: embedding_euclidean_distance --
Accuracies: [0.737]
Average Accuracy: 0.7370
F1 Scores: [0.667041778027852]
Average F1 Score: 0.6670
Best Parameters per Repetition:
 Repetition 1: {'classifier__C': 10, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
Best CV Scores per Repetition:
 Repetition 1: 0.7400
Classification Reports:

--- Repetition 1 ---
              precision    recall  f1-score   support

           0       0.69      0.90      0.78      4732
           1       0.83      0.56      0.67      4268

    accuracy                           0.74      9000
   macro avg       0.76      0.73      0.72      9000
weighted avg       0.76      0.74      0.73      9000

'''