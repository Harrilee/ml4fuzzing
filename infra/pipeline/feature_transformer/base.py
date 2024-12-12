from abc import ABC, abstractmethod
from typing import Any

class FeatureTransformer(ABC):
    """Base class for all feature transformers."""
    
    @abstractmethod
    def fit(self, X: Any) -> 'FeatureTransformer':
        """Fit the transformer to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Transform the features."""
        pass
        
    def fit_transform(self, X: Any) -> Any:
        """Fit the transformer and transform the data."""
        return self.fit(X).transform(X) 