import numpy as np
from typing import Union, Any, Tuple, Optional
import pandas as pd
from .base import DataPicker

class RandomDataPicker(DataPicker):
    """
    A data picker that randomly selects a specified number of samples.
    """
    
    def __init__(self, n_samples: int, random_state: int = None):
        """
        Initialize the random data picker.
        
        Args:
            n_samples: Number of samples to pick
            random_state: Random state for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        
    def pick(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Any, Any]:
        """
        Randomly pick n_samples from the input data.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Target values (Series, numpy array, or None)
            
        Returns:
            Tuple of (X_selected, y_selected) containing the randomly selected subset
        """
        rng = np.random.RandomState(self.random_state)
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            n_samples = min(self.n_samples, len(X))
            indices = rng.choice(len(X), size=n_samples, replace=False)
            X_selected = X.iloc[indices]
        elif isinstance(X, np.ndarray):
            n_samples = min(self.n_samples, X.shape[0])
            indices = rng.choice(X.shape[0], size=n_samples, replace=False)
            X_selected = X[indices]
        else:
            raise ValueError("X must be either pandas DataFrame/Series or numpy array")
            
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_selected = y.iloc[indices]
            elif isinstance(y, np.ndarray):
                y_selected = y[indices]
            else:
                raise ValueError("y must be either pandas DataFrame/Series or numpy array")
        else:
            y_selected = None
        return X_selected, y_selected