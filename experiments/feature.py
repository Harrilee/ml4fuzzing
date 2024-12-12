from collections import defaultdict
import sys
import os
import numpy as np

# Add the parent directory to the system path BEFORE any other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from infra.pipeline.data_picker.random_picker import RandomDataPicker
from infra.pipeline.feature_transformer.word_count_transformer import WordCountTransformer
from infra.pipeline.feature_transformer.tfidf_trace_transformer import TFIDFTraceTransformer
from infra.training.data_loader import DataLoader
from infra.training.trainer import Trainer
from infra.pipeline.pipeline_builder import PipelineBuilder
from sklearn.linear_model import LogisticRegression

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

# Initialize pipeline builder with logistic regression classifier
print("\nBuilding pipeline...")
pipeline_builder = PipelineBuilder()

# Initialize trainer
print("\nInitializing experiment...")

res = defaultdict(list)
feature_transformers = [WordCountTransformer(), TFIDFTraceTransformer()]
project_names = [d[2] for d in datasets]

for feature_transformer in feature_transformers:
    print(f" \n----- {feature_transformer.__class__.__name__} -----")
    cur_res = defaultdict(list)
    for i in range(1):
        pipeline = pipeline_builder\
            .set_data_picker(RandomDataPicker(n_samples=50, random_state=42+i))\
            .set_feature_transformer(feature_transformer)\
            .set_classifier(LogisticRegression(max_iter=5000))\
            .build()
        trainer = Trainer(pipeline=pipeline, test_size=0.3, random_state=42)
        for X, y, project_name, mutation_index in datasets:
            print(f"\nProcessing dataset: {project_name}")
            cur_res[project_name].append(np.round(trainer.train(X, y)['f1_score'], 4))
    for project_name, results in cur_res.items():
        res[feature_transformer.__class__.__name__].append(np.round(sum(results)/len(results), 4))

for feature_transformer, results in res.items():
    print(f"\nResults for {feature_transformer}:")
    for project_name, result in zip(project_names, results):
        print(f"  {project_name}: {result}")
    
    