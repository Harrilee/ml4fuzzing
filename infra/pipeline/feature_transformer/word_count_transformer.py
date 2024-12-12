from typing import Any, Union
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from .base import FeatureTransformer

class WordCountTransformer(FeatureTransformer):
    """
    A transformer that converts execution traces into word count vectors.
    Includes additional parameters like minimum document frequency.
    """
    
    def __init__(self, min_df: int = 3, max_features: int = 600):
        """
        Initialize the word count transformer.
        
        Args:
            min_df: Minimum document frequency (ignore terms that appear in fewer documents)
            max_features: Maximum number of features to create
        """
        self.min_df = min_df
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_features=max_features
        )
    
    def _prepare_input(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Helper method to prepare input data."""
        # If input is DataFrame, get the exec_trace column
        if isinstance(X, pd.DataFrame):
            if 'exec_trace' not in X.columns:
                raise ValueError("DataFrame must contain 'exec_trace' column")
            X = X['exec_trace']
        
        # Convert list of traces to space-separated strings
        return X.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        
    def fit(self, X: Union[pd.DataFrame, pd.Series]) -> 'WordCountTransformer':
        """
        Fit the vectorizer to the data.
        
        Args:
            X: Input data containing execution traces
               Can be a DataFrame with 'exec_trace' column or a Series of traces
        """
        traces = self._prepare_input(X)
        self.vectorizer.fit(traces)
        return self
        
    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Any:
        """
        Transform execution traces into word count vectors.
        
        Args:
            X: Input data containing execution traces
               Can be a DataFrame with 'exec_trace' column or a Series of traces
            
        Returns:
            Sparse matrix of word count vectors
        """
        traces = self._prepare_input(X)
        return self.vectorizer.transform(traces)
    
    def get_feature_names(self) -> list:
        """Get the names of the features created by the vectorizer."""
        return self.vectorizer.get_feature_names_out() 