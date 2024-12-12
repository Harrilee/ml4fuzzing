import numpy as np
from typing import Union, Any, Tuple, Optional
import pandas as pd
from sklearn.metrics import pairwise_distances
from textdistance import Jaccard
from .base import DataPicker

class JaccardDataPicker(DataPicker):
    """
    A data picker that selects samples based on Jaccard distance.
    """
    
    def __init__(self, n_samples: int, random_state: int = None):
        """
        Initialize the Jaccard distance data picker.
        
        Args:
            n_samples: Total number of samples to pick
            random_state: Random state for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        
    def pick(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Any, Any]:
        """
        Pick samples based on Jaccard distance.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Target values (Series, numpy array, or None)
            
        Returns:
            Tuple of (X_selected, y_selected) containing the selected subset
        """
        
        # Check if the input is text and apply Count Vectorization
        if isinstance(X, pd.DataFrame) and X.select_dtypes(include=[object]).shape[1] > 0:
            # Assuming the text data is in the first column
            text_data = X.iloc[:, 0].astype(str).to_numpy()
        elif isinstance(X, np.ndarray) and X.dtype == object:
            # Assuming the text data is in a 1D numpy array
            text_data = X.astype(str)
        else:
            raise ValueError("X must be either pandas DataFrame or numpy array with text data")

        # Initialize the Jaccard distance calculator
        jaccard = Jaccard()

        # Define a custom distance function using textdistance
        def jaccard_distance(u, v):
            return jaccard.distance(u[0], v[0])

        # Calculate pairwise Jaccard distances using the custom function
        distance_matrix = pairwise_distances(text_data.reshape(-1, 1), metric=jaccard_distance)
        
        # Select samples based on Jaccard distance
        selected_indices = np.argsort(distance_matrix.sum(axis=1))[:self.n_samples]
        
        X_selected = X.iloc[selected_indices] if isinstance(X, pd.DataFrame) else X[selected_indices]
        
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_selected = y.iloc[selected_indices]
            elif isinstance(y, np.ndarray):
                y_selected = y[selected_indices]
            else:
                raise ValueError("y must be either pandas DataFrame/Series or numpy array")
        else:
            y_selected = None
        
        return X_selected, y_selected
