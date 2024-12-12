from typing import Optional, Union
from sklearn.base import BaseEstimator
import tensorflow as tf
import torch
from .data_picker.base import DataPicker
from .feature_transformer.base import FeatureTransformer

class DeepLearningAdapter(BaseEstimator):
    """
    Adapter class to make deep learning models (PyTorch and TensorFlow)
    compatible with scikit-learn interface.
    """
    def __init__(self, model):
        self.model = model
        self.is_torch = isinstance(model, torch.nn.Module)
        if self.is_torch:
            self.model.train()

    def fit(self, X, y=None):
        """
        Placeholder for fit method. Deep learning models should be trained
        separately using their respective training loops.
        """
        # Set model to evaluation mode for PyTorch models
        if self.is_torch:
            self.model.eval()
        return self

    def predict(self, X):
        """
        Make predictions using the deep learning model.
        """
        if self.is_torch:
            with torch.no_grad():
                # Convert input to torch tensor if it isn't already
                if not isinstance(X, torch.Tensor):
                    X = torch.FloatTensor(X)
                outputs = self.model(X)
                # Return predictions as numpy array
                return outputs.numpy()
        else:
            # Assume TensorFlow model
            return self.model.predict(X)

class CustomPipeline:
    """
    Custom pipeline that handles data picking, feature transformation
    and classification steps directly.
    """
    def __init__(
        self,
        data_picker: Optional[DataPicker] = None,
        feature_transformer: Optional[FeatureTransformer] = None,
        classifier: Optional[Union[BaseEstimator, 'torch.nn.Module', 'tf.keras.Model']] = None
    ):
        self.data_picker = data_picker
        self.feature_transformer = feature_transformer
        
        # Handle classifier based on its type
        if isinstance(classifier, (torch.nn.Module, tf.keras.Model)):
            self.classifier = DeepLearningAdapter(classifier)
        else:
            self.classifier = classifier

    def fit(self, X, y=None):
        """Fit the pipeline on the data"""
        # Apply data picking if available
        if self.data_picker:
            X, y = self.data_picker.pick(X, y)
        
        # Apply feature transformation if available
        if self.feature_transformer:
            # First fit and transform the training data
            X = self.feature_transformer.fit_transform(X)
            
        print('fit:', X.shape)
        self.classifier.fit(X, y)
            
        return self

    def predict(self, X):
        """Make predictions using the pipeline"""
        # Apply feature transformation if available
        if self.feature_transformer:
            X = self.feature_transformer.transform(X)
            
        # Make predictions if classifier is available
        if self.classifier:
            return self.classifier.predict(X)
        return X

    def transform(self, X):
        """Transform the data using the pipeline"""
        if self.feature_transformer:
            return self.feature_transformer.transform(X)
        return X

class PipelineBuilder:
    """
    A flexible pipeline builder that combines data picking, feature transformation,
    and classification steps.
    """
    def __init__(
        self,
        data_picker: Optional[DataPicker] = None,
        feature_transformer: Optional[FeatureTransformer] = None,
        classifier: Optional[Union[BaseEstimator, 'torch.nn.Module', 'tf.keras.Model']] = None
    ):
        self.data_picker = data_picker
        self.feature_transformer = feature_transformer
        self.classifier = classifier
        
    def set_data_picker(self, data_picker: DataPicker) -> 'PipelineBuilder':
        """Set the data picker component"""
        self.data_picker = data_picker
        return self
        
    def set_feature_transformer(self, feature_transformer: FeatureTransformer) -> 'PipelineBuilder':
        """Set the feature transformer component"""
        self.feature_transformer = feature_transformer
        return self
        
    def set_classifier(self, classifier: Union[BaseEstimator, 'torch.nn.Module', 'tf.keras.Model']) -> 'PipelineBuilder':
        """Set the classifier component"""
        self.classifier = classifier
        return self

    def build(self) -> CustomPipeline:
        """
        Build and return a CustomPipeline with the configured components.
        """
        return CustomPipeline(
            data_picker=self.data_picker,
            feature_transformer=self.feature_transformer,
            classifier=self.classifier
        )