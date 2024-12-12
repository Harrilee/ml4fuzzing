from typing import List, Any
from abc import ABC, abstractmethod

class DataPicker(ABC):
    """Abstract base class for data picking strategies"""
    @abstractmethod
    def pick(self, data: List[Any]) -> List[Any]:
        """Pick a subset of data from the input list"""
        pass

class FeatureTransformer(ABC):
    """Abstract base class for feature transformation strategies"""
    @abstractmethod
    def transform(self, data: List[Any]) -> List[Any]:
        """Transform the input data into features"""
        pass 