import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional

class DataLoader:
    """Loads and preprocesses execution trace data from log files."""
    
    def __init__(self, base_dir: str):
        """
        Initialize the DataLoader.
        
        Args:
            base_dir: Base directory containing the log files
        """
        self.base_dir = base_dir

    def get_logs(self, logs_dir: str, mutation_index: int) -> List[Dict]:
        """
        Load logs for a specific mutation index.
        
        Args:
            logs_dir: Directory containing log files
            mutation_index: Index of the mutation to load
            
        Returns:
            List of log dictionaries
        """
        files = os.listdir(logs_dir)
        logs = []
        
        for file_name in files:
            if file_name.startswith(f"mutation{mutation_index}_"):
                with open(os.path.join(logs_dir, file_name), "r") as f:
                    logs.append(json.load(f))
                    
        print(f"Number of logs loaded: {len(logs)}")
        return logs

    def combine_logs(self, logs: List[Dict]) -> pd.DataFrame:
        """
        Combine multiple logs into a single DataFrame.
        
        Args:
            logs: List of log dictionaries
            
        Returns:
            Combined DataFrame
        """
        combined_logs = [log for log in logs if isinstance(log, dict)]
        df = pd.DataFrame(combined_logs)
        return df

    def load_data(
        self,
        logs_subdir: str,
        mutation_index: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess data for training.
        
        Args:
            logs_subdir: Subdirectory containing log files
            mutation_index: Index of the mutation to load
            
        Returns:
            Tuple of (X, y) where X is the feature DataFrame and y is the target series
        """
        logs_dir = os.path.join(self.base_dir, logs_subdir)
        print(f"\nLoading data from: '{logs_dir}'")
        print(f"Mutation Index: {mutation_index}")

        # Get project name from path
        project_name = logs_subdir.split('/')[0]
        print(f"Project: {project_name}")

        # Load and combine logs
        logs = self.get_logs(logs_dir, mutation_index)
        if not logs:
            raise ValueError(f"No logs found for mutation index {mutation_index} in '{logs_dir}'")

        df = self.combine_logs(logs)
        df = df.dropna(subset=['exec_trace', 'verdict'])
        
        # Prepare features and target
        X = df[['exec_trace']]
        y = df['verdict'].apply(lambda x: 1 if x.lower() == 'pass' else 0)
        
        return X, y

    def load_multiple_datasets(
        self,
        logs_subdirs_to_mutations: Dict[str, List[int]]
    ) -> List[Tuple[pd.DataFrame, pd.Series, str, int]]:
        """
        Load multiple datasets specified by a dictionary.
        
        Args:
            logs_subdirs_to_mutations: Dictionary mapping subdirectories to lists of mutation indices
            
        Returns:
            List of tuples (X, y, project_name, mutation_index)
        """
        datasets = []
        
        for logs_subdir, mutation_indices in logs_subdirs_to_mutations.items():
            project_name = logs_subdir.split('/')[0]
            
            for mutation_index in mutation_indices:
                try:
                    X, y = self.load_data(logs_subdir, mutation_index)
                    datasets.append((X, y, project_name, mutation_index))
                except Exception as e:
                    print(f"Error loading data for {project_name}, mutation {mutation_index}: {str(e)}")
                    continue
                    
        return datasets 