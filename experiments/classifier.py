from collections import defaultdict
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Add the parent directory to the system path BEFORE any other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infra.pipeline.data_picker.random_picker import RandomDataPicker
from infra.pipeline.feature_transformer.word_count_transformer import WordCountTransformer
from infra.training.data_loader import DataLoader
from infra.training.trainer import Trainer
from infra.pipeline.pipeline_builder import PipelineBuilder

# Define base directory and datasets to process
base_dir = "fuzz_test"

# Define which logs to process
logs_subdirs_to_mutations = {
    "textdistance/test_DamerauLevenshtein/logs": [2],
    "dateutil/test_date_parse/logs": [3],
}

def create_classifier(name):
    """Create a new classifier instance based on name"""
    classifiers = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            warm_start=False  # Ensure fresh start for each fit
        ),
        'SVC': lambda: SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        ),
        'MultinomialNB': lambda: MultinomialNB(),
        'DecisionTree': lambda: DecisionTreeClassifier(random_state=42),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': lambda: LogisticRegression(max_iter=5000)
    }

    return classifiers[name]()

# List of classifier names to test
classifier_names = [
    'RandomForest', 
    'SVC', 
    'MultinomialNB',
    'DecisionTree', 
    'KNN', 
    'LogisticRegression'
]

# Initialize data loader
print("Initializing DataLoader...")
data_loader = DataLoader(base_dir=base_dir)

# Load all datasets
print("\nLoading datasets...")
datasets = data_loader.load_multiple_datasets(logs_subdirs_to_mutations)

# Initialize pipeline builder
print("\nBuilding pipeline...")
pipeline_builder = PipelineBuilder()

# Initialize experiment results storage
results = defaultdict(list)
feature_transformer = WordCountTransformer()  # Use WordCountTransformer as default
project_names = [d[2] for d in datasets]


for classifier_name in classifier_names:
    print(f"\n----- Testing {classifier_name} -----")
    
    cur_res = defaultdict(list)
    for i in range(1):  # You can increase this for multiple runs
        try:
            # Create a fresh classifier instance for each run
            classifier = create_classifier(classifier_name)
            
            pipeline = pipeline_builder\
                .set_data_picker(RandomDataPicker(n_samples=50, random_state=42+i))\
                .set_feature_transformer(feature_transformer)\
                .set_classifier(classifier)\
                .build() 
                
            trainer = Trainer(pipeline=pipeline, test_size=0.3, random_state=42)
            
            for X, y, project_name, mutation_index in datasets:
                print(f"Processing dataset: {project_name}")
                try:
                    metrics = trainer.train(X, y)
                    score = metrics.get('f1_score', 0.0)  # Use get() to handle missing metrics
                    cur_res[project_name].append(np.round(score, 4))
                except Exception as e:
                    print(f"Error processing {project_name} with {classifier_name}: {str(e)}")
                    cur_res[project_name].append(0.0)  # Record failed attempt as 0
                    continue  # Continue with next dataset
                    
        except Exception as e:
            print(f"Error with classifier {classifier_name}: {str(e)}")
            continue  # Continue with next classifier
    
    # Average results across runs (only if we have results)
    for project_name, scores in cur_res.items():
        if scores:  # Check if we have any valid scores
            avg_score = np.round(sum(scores)/len(scores), 4)
            results[classifier_name].append(avg_score)
        else:
            results[classifier_name].append(0.0)

# Print results
print("\n========== FINAL RESULTS ==========")
for classifier_name, scores in results.items():
    print(f"\n{classifier_name}:")
    for project_name, score in zip(project_names, scores):
        print(f"  {project_name}: {score}")

# Calculate and print the best performing model
print("\nBest model:")
best_avg_score = -1
best_classifier = None

for classifier_name, scores in results.items():
    if scores:  # Check if we have any scores for this classifier
        avg_score = np.mean(scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_classifier = classifier_name

if best_classifier:
    print(f"  {best_classifier} (average F1: {np.round(best_avg_score, 4)})")
else:
    print("  No successful models found") 