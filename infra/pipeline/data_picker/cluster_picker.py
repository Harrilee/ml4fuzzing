import numpy as np
from typing import Union, Any, Tuple, Optional
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from .base import DataPicker

class ClusterDataPicker(DataPicker):
    """
    A data picker that selects samples based on clustering.
    """
    
    def __init__(self, n_samples: int, random_state: int = None):
        """
        Initialize the cluster data picker.
        
        Args:
            n_samples: Total number of samples to pick
            n_clusters: Number of clusters to form
            random_state: Random state for reproducibility
        """
        self.n_samples = n_samples
        self.n_clusters = n_samples
        self.random_state = random_state
        
    def pick(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[Any, Any]:
        """
        Pick samples based on clustering.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Target values (Series, numpy array, or None)
            
        Returns:
            Tuple of (X_selected, y_selected) containing the selected subset
        """
        
        # Check if the input is text and apply Count Vectorization
        if isinstance(X, pd.DataFrame) and X.select_dtypes(include=[object]).shape[1] > 0:
            # Assuming the text data is in the first column
            text_data = X.iloc[:, 0].astype(str)
            vectorizer = CountVectorizer()
            X_data = vectorizer.fit_transform(text_data).toarray()
        elif isinstance(X, np.ndarray) and X.dtype == object:
            # Assuming the text data is in a 1D numpy array
            vectorizer = CountVectorizer()
            X_data = vectorizer.fit_transform(X.astype(str)).toarray()
        else:
            # Existing numeric data handling
            if isinstance(X, pd.DataFrame):
                X_data = X.select_dtypes(include=[np.number]).values
            elif isinstance(X, np.ndarray):
                X_data = X
            else:
                raise ValueError("X must be either pandas DataFrame or numpy array")

        # Flatten and reshape if necessary
        if X_data.ndim == 1:
            X_data = X_data.reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_data)
        
        # Calculate number of samples per cluster
        samples_per_cluster = max(1, self.n_samples // self.n_clusters)
        
        selected_indices = []
        for cluster in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 0:
                rng = np.random.RandomState(self.random_state)
                selected_indices.extend(rng.choice(cluster_indices, size=min(samples_per_cluster, len(cluster_indices)), replace=False))
        
        print(f"Selected indices: {selected_indices}")
        
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