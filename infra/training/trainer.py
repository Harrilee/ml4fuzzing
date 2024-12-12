import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import Dict, Any
import pandas as pd

class Trainer:
    """Trains and evaluates machine learning pipelines with specified data."""
    
    def __init__(self, pipeline: Any, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            pipeline: The sklearn pipeline to train
            test_size: Proportion of dataset to include in the test split
            random_state: Random state for reproducibility
        """
        self.pipeline = pipeline
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Prepare data by performing train-test split.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Train Size: {len(self.X_train)}")
        print(f"Test Size: {len(self.X_test)}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the pipeline and evaluate performance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare the data
        self.prepare_data(X, y)
        
        # Train the model
        print("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.pipeline.predict(self.X_test)
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred, average='macro', zero_division=0),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        
        # Print results
        print("\n=== Results ===")
        print(f"Test Set Accuracy: {results['accuracy']:.4f}")
        print(f"Test Set F1 Score: {results['f1_score']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results 