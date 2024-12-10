# %%
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

pd.set_option('mode.chained_assignment', None)


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


# Note: Not in use for now
# Train model, hyperparameter tuning with GridSearchCV
def train_model(classifier, X_train, y_train, param_grid):
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    return grid_search


# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return {
        'accuracy': accuracy,
        'classification_report': report
    }


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")


def get_training_dataset(X, y, vectorization):
    if vectorization == 'tfidf':
        # Combine the list of traces into a single string for TF-IDF vectorization
        X.loc[:, "exec_trace_combined"] = X["exec_trace"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X['exec_trace_combined'], y, test_size=0.2, random_state=42, stratify=y
        )

        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=0.03,
            max_df=0.9,
            max_features=600,
        )

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        return X_train_tfidf, y_train, X_test_tfidf, y_test
    # TODO: Add more vectorization methods as needed
    else:
        raise ValueError(f"Unknown Vectorization Method in get_processed_dataset: {vectorization}")


class MutationModelTrainer:
    def __init__(self, base_dir, logs_subdirs_to_mutations, classifier_list, model_save_dir):
        self.base_dir = base_dir
        self.logs_subdirs_to_mutations = logs_subdirs_to_mutations
        self.classifier_list = classifier_list
        self.model_save_dir = model_save_dir
        self.results = {}

        os.makedirs(self.model_save_dir, exist_ok=True)

    def process_mutation(self, logs_subdir, mutation_index):
        logs_dir = os.path.join(self.base_dir, logs_subdir)
        print(f"\nProcessing Logs Subdir: '{logs_subdir}', Mutation Index: {mutation_index}")

        project_name = logs_subdir.split('/')[0]
        print(f"Project Name: {project_name}")

        logs = get_logs(logs_dir, mutation_index)

        if not logs:
            print(f"No logs found for mutation index {mutation_index} in '{logs_subdir}'. Skipping.")
            return

        df = combine_logs(logs)
        df = df.dropna(subset=['exec_trace', 'verdict'])

        y = df['verdict'].apply(lambda x: 1 if x.lower() == 'pass' else 0)
        X = df[['exec_trace']]

        X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf = get_training_dataset(X, y, 'tfidf')

        # Train the model
        for classifier in self.classifier_list:
            # grid_search = train_model(classifier, X_train_tfidf, y_train_tfidf)
            classifier.fit(X_train_tfidf, y_train_tfidf)

            # Evaluate the best model
            evaluation = evaluate_model(classifier, X_test_tfidf, y_test_tfidf)

            # Save the best model
            model_name = classifier.__class__.__name__
            print(f"--------------------Model used: {model_name}--------------------")
            # model_filename = os.path.join(self.model_save_dir,
            #                               f"{project_name}_mutation_{mutation_index}_{model_name}.pkl")
            # save_model(grid_search.best_estimator_, model_filename)

            self.results[mutation_index] = {
                # 'best_params': grid_search.best_params_,
                # 'best_score': grid_search.best_score_,
                'test_accuracy': evaluation['accuracy'],
                'classification_report': evaluation['classification_report']
            }

            # print("Best Parameters:")
            # print(grid_search.best_params_)
            # print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
            print(f"Test Set Accuracy: {evaluation['accuracy']:.4f}")
            print("Classification Report:")
            print(evaluation['classification_report'])

    def train_all(self):
        for logs_subdir, mutation_indices in self.logs_subdirs_to_mutations.items():
            for mutation_index in mutation_indices:
                self.process_mutation(logs_subdir, mutation_index)

    def get_results(self):
        return self.results


def main():
    base_dir = "./fuzz_test"

    # Subdirectory path: [mutation indices]
    logs_subdirs_to_train = {
        "textdistance/test_DamerauLevenshtein/logs": [1, 2, 3, 4, 5],
        "dateutil/test_date_parse/logs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    model_save_dir = "./ml/models"

    classifier_list = [
        SVC(random_state=0),
        RandomForestClassifier(random_state=0)
    ]

    trainer = MutationModelTrainer(
        base_dir=base_dir,
        logs_subdirs_to_mutations=logs_subdirs_to_train,
        classifier_list = classifier_list,
        model_save_dir=model_save_dir
    )

    trainer.train_all()

    results = trainer.get_results()


if __name__ == "__main__":
    main()
