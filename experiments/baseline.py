from collections import defaultdict
import sys
import os

import numpy as np
from sklearn.naive_bayes import MultinomialNB

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
    "textdistance/test_DamerauLevenshtein/logs": range(1,6),
    "dateutil/test_date_parse/logs": range(1, 11),
}

# Initialize data loader
print("Initializing DataLoader...")
data_loader = DataLoader(base_dir=base_dir)

# Load all datasets
print("\nLoading datasets...")
datasets = data_loader.load_multiple_datasets(logs_subdirs_to_mutations)
print(len(datasets))

# Initialize pipeline builder with logistic regression classifier
print("\nBuilding pipeline...")
pipeline_builder = PipelineBuilder()

# Initialize trainer
print("\nInitializing experiment...")

res = defaultdict(list)

for i in range(3):
    pipeline = pipeline_builder\
        .set_data_picker(RandomDataPicker(n_samples=50, random_state=42+i))\
        .set_feature_transformer(WordCountTransformer())\
        .set_classifier(MultinomialNB())\
        .build()
    trainer = Trainer(pipeline=pipeline, test_size=0.3, random_state=42)
    for X, y, project_name, mutation_index in datasets:
        print(f"\nProcessing dataset: {project_name}")
        res[project_name].append(np.round(trainer.train(X, y)['f1_score'], 4))

for project_name, results in res.items():
    print(f"\nResults for {project_name}: {np.round(sum(results)/len(results), 4)}")
    
    