from typing import Any, Union
import pandas as pd
from .base import FeatureTransformer

class BooleanTraceTransformer(FeatureTransformer):
    """
    A transformer that converts execution traces into Boolean Matrix.
    Uses term frequency-inverse document frequency to weight the importance of terms.
    """

    def __init__(self):
        """
        Initialize the boolean transformer.
        """
        self.unique_strings = set()

    def _prepare_input(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Helper method to prepare input data."""
        # If input is DataFrame, get the exec_trace column
        if isinstance(X, pd.DataFrame):
            if 'exec_trace' not in X.columns:
                raise ValueError("DataFrame must contain 'exec_trace' column")
            X = X['exec_trace']

        # Convert list of traces to space-separated strings
        return X.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    def fit(self, X: Union[pd.DataFrame, pd.Series]) -> 'BooleanTraceTransformer':
        """
        Fit the vectorizer to the data.

        Args:
            X: Input data containing execution traces
               Can be a DataFrame with 'exec_trace' column or a Series of traces
        """
        traces = self._prepare_input(X)

        for trace in traces:
            self.unique_strings.update(trace.split())

        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> Any:
        """
        Transform execution traces into TF-IDF vectors.

        Args:
            X: Input data containing execution traces
               Can be a DataFrame with 'exec_trace' column or a Series of traces

        Returns:
            Sparse matrix of TF-IDF vectors
        """
        boolean_matrix = []
        traces = self._prepare_input(X)

        for trace in traces:
            trace_words = set(trace.split())
            row = [string in trace_words for string in self.unique_strings]
            boolean_matrix.append(row)

        test_boolean_df = pd.DataFrame(boolean_matrix, columns=list(self.unique_strings))

        return test_boolean_df

    def get_feature_names(self) -> list:
        """Get the names of the features created by the vectorizer."""
        return list(self.unique_strings)