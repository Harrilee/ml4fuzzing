from abc import ABC, abstractmethod
from typing import Any

class DataPicker(ABC):
    """Base class for all data pickers."""
    
    @abstractmethod
    def pick(self, X: Any) -> Any:
        """Pick and transform the data."""
        pass 