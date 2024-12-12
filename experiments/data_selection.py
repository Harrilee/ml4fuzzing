from collections import defaultdict
import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

# Add the parent directory to the system path BEFORE any other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infra.pipeline.data_picker.random_picker import RandomDataPicker
from infra.pipeline.data_picker.cluster_picker import ClusterDataPicker
from infra.pipeline.data_picker.edit_distance_picker import EditDistanceDataPicker
from infra.pipeline.feature_transformer.word_count_transformer import WordCountTransformer
from infra.training.data_loader import DataLoader
from infra.training.trainer import Trainer
from infra.pipeline.pipeline_builder import PipelineBuilder
from textdistance import DamerauLevenshtein

# Define base directory and datasets to process
base_dir = "fuzz_test"

# Define which logs to process
logs_subdirs_to_mutations = {
    "textdistance/test_DamerauLevenshtein/logs": [2],
    "dateutil/test_date_parse/logs": [3],
}

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

# Define data pickers to test
data_pickers = {
    # 'RandomDataPicker': lambda: RandomDataPicker(n_samples=50, random_state=42),
    # 'ClusterDataPicker': lambda: ClusterDataPicker(n_samples=50, random_state=42),
    'EditDistancePicker': lambda: EditDistanceDataPicker(n_samples=50, random_state=42)
}

# Use Logistic Regression as the classifier
classifier = LogisticRegression(max_iter=5000)

for picker_name, picker_factory in data_pickers.items():
    print(f"\n----- Testing with {picker_name} -----")
    
    cur_res = defaultdict(list)
    for i in range(1):  # You can increase this for multiple runs
        try:
            # Create a fresh data picker instance for each run
            data_picker = picker_factory()
            
            pipeline = pipeline_builder\
                .set_data_picker(data_picker)\
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
                    print(f"Error processing {project_name} with {picker_name}: {str(e)}")
                    cur_res[project_name].append(0.0)  # Record failed attempt as 0
                    continue  # Continue with next dataset
                    
        except Exception as e:
            print(f"Error with data picker {picker_name}: {str(e)}")
            continue  # Continue with next data picker
    
    # Average results across runs (only if we have results)
    for project_name, scores in cur_res.items():
        if scores:  # Check if we have any valid scores
            avg_score = np.round(sum(scores)/len(scores), 4)
            results[picker_name].append(avg_score)
        else:
            results[picker_name].append(0.0)

# Print results
print("\n========== FINAL RESULTS ==========")
for picker_name, scores in results.items():
    print(f"\n{picker_name}:")
    for project_name, score in zip(project_names, scores):
        print(f"  {project_name}: {score}")

# Calculate and print the best performing data picker
print("\nBest data picker:")
best_avg_score = -1
best_picker = None

for picker_name, scores in results.items():
    if scores:  # Check if we have any scores for this data picker
        avg_score = np.mean(scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_picker = picker_name

if best_picker:
    print(f"  {best_picker} (average F1: {np.round(best_avg_score, 4)})")
else:
    print("  No successful data pickers found") 