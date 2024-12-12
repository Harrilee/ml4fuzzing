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

    samples_per_cluster = max(1, sample_size // n_clusters)

    # Perform sampling within each cluster
    sampled_df = df.groupby('cluster').apply(
        lambda x: x.sample(n=min(len(x), samples_per_cluster), random_state=random_state)
    )

    # Flatten MultiIndex and preserve original indices
    sampled_df = sampled_df.reset_index(level=0, drop=True)

    # Calculate remaining samples to reach sample_size
    current_sample_size = len(sampled_df)
    if current_sample_size < sample_size:
        remaining_indices = list(set(X.index) - set(sampled_df.index))
        additional_samples = sample_size - current_sample_size
        additional_df = X.loc[remaining_indices].sample(n=additional_samples, random_state=random_state)
        sampled_df = pd.concat([sampled_df, additional_df], ignore_index=False)

    # Align sampled_df indices with y
    sampled_indices = sampled_df.index

    # Return the sampled exec_trace and corresponding y
    return sampled_df[['exec_trace']], y.loc[sampled_indices]


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
            print(f"\rPrecomputing edit distance: {i + 1}/{len(X_inputs)}", end='')
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

    token_sets = X_input.apply(
        lambda trace: set(' '.join(trace).split()) if isinstance(trace, list) else set(trace.split()))

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
            print(f"\rPrecomputing Jaccard distance: {i + 1}/{len(token_sets)}", end='')
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


# Simple Active Learner Class
class SimpleActiveLearner:
    def __init__(self, pipeline, X_pool, y_pool, X_test, y_test, initial_size=100, query_size=10, iterations=10, random_state=42):
        self.pipeline = pipeline
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.initial_size = initial_size
        self.query_size = query_size
        self.iterations = iterations
        self.random_state = random_state
        self.labeled_X = None
        self.labeled_y = None
        self.results = []

    def initialize(self):
        np.random.seed(self.random_state)
        self.X_pool = self.X_pool.reset_index(drop=True)
        self.y_pool = self.y_pool.reset_index(drop=True)
        initial_indices = np.random.choice(range(len(self.X_pool)), size=self.initial_size, replace=False)
        self.labeled_X = self.X_pool.iloc[initial_indices]
        self.labeled_y = self.y_pool.iloc[initial_indices]
        self.X_pool = self.X_pool.drop(initial_indices).reset_index(drop=True)
        self.y_pool = self.y_pool.drop(initial_indices).reset_index(drop=True)

    def uncertainty_sampling(self):
        self.pipeline.fit(self.labeled_X['exec_trace'], self.labeled_y)
        probs = self.pipeline.predict_proba(self.X_pool['exec_trace'])
        uncertainty = 1 - np.max(probs, axis=1)
        query_indices = uncertainty.argsort()[-self.query_size:]
        return query_indices

    def run(self):
        self.initialize()
        for it in range(1, self.iterations + 1):
            print(f"\n--- Active Learning Iteration {it}/{self.iterations} ---")
            if len(self.X_pool) == 0:
                print("No more samples in the pool.")
                break
            query_indices = self.uncertainty_sampling()
            queried_X = self.X_pool.iloc[query_indices]
            queried_y = self.y_pool.iloc[query_indices]
            self.labeled_X = pd.concat([self.labeled_X, queried_X], ignore_index=True)
            self.labeled_y = pd.concat([self.labeled_y, queried_y], ignore_index=True)
            self.X_pool = self.X_pool.drop(query_indices).reset_index(drop=True)
            self.y_pool = self.y_pool.drop(query_indices).reset_index(drop=True)
            self.pipeline.fit(self.labeled_X['exec_trace'], self.labeled_y)
            y_pred = self.pipeline.predict(self.X_test['exec_trace'])
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            self.results.append({'accuracy': accuracy, 'f1_score': f1, 'report': report})
            print(f"Accuracy after iteration {it}: {accuracy:.4f}")
            print(f"F1 Score after iteration {it}: {f1:.4f}")
            print("Classification Report:")
            print(report)
        return self.results


# Main training class
class MutationModelTrainer:
    def __init__(self, base_dir, logs_subdirs_to_mutations, param_grid, model_save_dir="models",
                 num_repeats=5, sample_size=500, seed_start=42, active_learning_iterations=10,
                 initial_training_size=100, query_size=10):
        self.base_dir = base_dir
        self.logs_subdirs_to_mutations = logs_subdirs_to_mutations
        self.param_grid = param_grid
        self.model_save_dir = model_save_dir
        self.num_repeats = num_repeats
        self.sample_size = sample_size
        self.seed_start = seed_start
        self.active_learning_iterations = active_learning_iterations
        self.initial_training_size = initial_training_size
        self.query_size = query_size
        self.results = {}

        os.makedirs(self.model_save_dir, exist_ok=True)

    def load_and_prepare_data(self, logs_subdir, mutation_index):
        logs_dir = os.path.join(self.base_dir, logs_subdir)
        print(f"\nProcessing Logs Subdir: '{logs_subdir}', Mutation Index: {mutation_index}")

        # Extract project name, assuming it's part of the directory structure
        project_name = logs_subdir.split(os.sep)[-3] if len(logs_subdir.split(os.sep)) >= 3 else logs_subdir.split('/')[
            -3]
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
            return cluster_based_sampling(X[['exec_trace']], y, self.sample_size, n_clusters=10,
                                          random_state=random_state)
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

    def active_learning_training(self, X_pool, y_pool, X_test, y_test, project_name, mutation_index):
        print("\n=== Starting Active Learning ===")

        # Initialize the Active Learner with a small labeled dataset
        learner = SimpleActiveLearner(
            pipeline=Pipeline([
                ('exec_transform', ExecTraceTransformer()),
                ('vectorizer', CountVectorizer()),
                ('feature_selection', SelectKBest(score_func=chi2, k=5)),
                ('classifier', LogisticRegression(max_iter=1000, random_state=self.seed_start))
            ]),
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            initial_size=self.initial_training_size,
            query_size=self.query_size,
            iterations=self.active_learning_iterations,
            random_state=self.seed_start
        )

        active_learning_results = learner.run()

        # Store Active Learning results
        if mutation_index not in self.results:
            self.results[mutation_index] = {}
        self.results[mutation_index]['active_learning'] = {
            'accuracies': [res['accuracy'] for res in active_learning_results],
            'average_accuracy': np.mean([res['accuracy'] for res in active_learning_results]),
            'f1_scores': [res['f1_score'] for res in active_learning_results],
            'average_f1_score': np.mean([res['f1_score'] for res in active_learning_results]),
            'classification_reports': [res['report'] for res in active_learning_results]
        }

    def process_mutation(self, logs_subdir, mutation_index, methods):
        X_train_full, X_test, y_train_full, y_test, project_name = self.load_and_prepare_data(logs_subdir,
                                                                                              mutation_index)
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

        # Active Learning Process
        self.active_learning_training(X_train_full, y_train_full, X_test, y_test, project_name, mutation_index)

    def train_all(self):
        # Define data selection methods to compare
        methods = [
            DataSelectionMethod.CLUSTER.value,
            DataSelectionMethod.RANDOM.value,
            DataSelectionMethod.STRATIFIED.value,

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

# Instantiate trainer with Active Learning parameters
trainer = MutationModelTrainer(
    base_dir=base_dir,
    logs_subdirs_to_mutations=logs_subdirs_to_train,
    param_grid=param_grid,
    model_save_dir=model_save_dir,
    num_repeats=5,
    sample_size=100,
    seed_start=42,
    active_learning_iterations=10,  # Number of Active Learning iterations
    initial_training_size=100,  # Initial labeled samples
    query_size=10  # Samples queried per iteration
)

# Train all models, including Active Learning
trainer.train_all()

# Retrieve and display results
results = trainer.get_results()

for mutation, methods in results.items():
    print(f"\n=== Mutation {mutation} ===")
    for method, metrics in methods.items():
        if method == 'active_learning':
            continue  # We'll handle Active Learning separately
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

    # Active Learning Results
    if 'active_learning' in methods:
        al_metrics = methods['active_learning']
        print(f"\n-- Active Learning --")
        print(f"Accuracies: {al_metrics['accuracies']}")
        print(f"Average Accuracy: {al_metrics['average_accuracy']:.4f}")
        print(f"F1 Scores: {al_metrics['f1_scores']}")
        print(f"Average F1 Score: {al_metrics['average_f1_score']:.4f}")
        print("Classification Reports:")
        for i, report in enumerate(al_metrics['classification_reports'], 1):
            print(f"\n--- Active Learning Iteration {i} ---")
            print(report)
