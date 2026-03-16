"""
Machine Learning Models for IoT Device Classification

This module contains various ML models and training utilities.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os
from typing import Dict, Any, Optional
from tqdm import tqdm
import time


class IoTClassifier:
    """Wrapper class for IoT device classification models."""
    
    AVAILABLE_MODELS = {
        'random_forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'description': 'Random Forest Classifier'
        },
        'decision_tree': {
            'class': DecisionTreeClassifier,
            'params': {
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'description': 'Decision Tree Classifier'
        },
        'xgboost': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            },
            'description': 'XGBoost Classifier'
        },
        'lightgbm': {
            'class': LGBMClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'description': 'LightGBM Classifier'
        },
        'knn': {
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5,
                'weights': 'distance',
                'n_jobs': -1
            },
            'description': 'K-Nearest Neighbors'
        },
        'svm': {
            'class': SVC,
            'params': {
                'kernel': 'rbf',
                'C': 1.0,
                'random_state': 42,
                'probability': True
            },
            'description': 'Support Vector Machine'
        },
        'logistic_regression': {
            'class': LogisticRegression,
            'params': {
                'max_iter': 1000,
                'random_state': 42,
                'n_jobs': -1
            },
            'description': 'Logistic Regression'
        },
        'mlp': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (128, 64, 32),
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.1
            },
            'description': 'Multi-Layer Perceptron'
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'description': 'Gradient Boosting Classifier'
        }
    }
    
    def __init__(self, model_name: str = 'random_forest', 
                 custom_params: Optional[Dict] = None):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model to use
            custom_params: Custom parameters to override defaults
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        model_config = self.AVAILABLE_MODELS[model_name]
        
        params = model_config['params'].copy()
        if custom_params:
            params.update(custom_params)
        
        self.model = model_config['class'](**params)
        self.description = model_config['description']
        self.is_fitted = False
        self.training_time = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'IoTClassifier':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self
        """
        print(f"\nTraining {self.description}...")
        
        start_time = time.time()
        
        if X_val is not None and y_val is not None:
            if hasattr(self.model, 'fit') and 'eval_set' in str(self.model.fit.__doc__ or ''):
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"{self.model_name} does not support probability predictions")
    
    def get_feature_importance(self, feature_names: list) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(sorted(zip(feature_names, importances), 
                             key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).mean(axis=0)
            return dict(sorted(zip(feature_names, importances), 
                             key=lambda x: x[1], reverse=True))
        return None
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'description': self.description,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'IoTClassifier':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded IoTClassifier instance
        """
        model_data = joblib.load(filepath)
        classifier = cls.__new__(cls)
        classifier.model = model_data['model']
        classifier.model_name = model_data['model_name']
        classifier.description = model_data['description']
        classifier.is_fitted = model_data['is_fitted']
        classifier.training_time = model_data['training_time']
        print(f"Model loaded from: {filepath}")
        return classifier


class ModelTrainer:
    """Utility class for training and comparing multiple models."""
    
    def __init__(self, models_to_train: Optional[list] = None):
        """
        Initialize the trainer.
        
        Args:
            models_to_train: List of model names to train (default: all)
        """
        if models_to_train is None:
            self.models_to_train = ['random_forest', 'decision_tree', 
                                   'xgboost', 'lightgbm', 'knn']
        else:
            self.models_to_train = models_to_train
        
        self.trained_models = {}
        
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> Dict[str, IoTClassifier]:
        """
        Train all specified models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("TRAINING MULTIPLE MODELS")
        print("="*60)
        print(f"Models to train: {self.models_to_train}")
        
        for model_name in tqdm(self.models_to_train, desc="Training models"):
            try:
                classifier = IoTClassifier(model_name)
                classifier.fit(X_train, y_train, X_val, y_val)
                self.trained_models[model_name] = classifier
            except Exception as e:
                print(f"\nError training {model_name}: {e}")
                continue
        
        print(f"\nSuccessfully trained {len(self.trained_models)} models")
        return self.trained_models
    
    def save_all(self, output_dir: str):
        """
        Save all trained models.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.trained_models.items():
            filepath = os.path.join(output_dir, f"{name}.joblib")
            model.save(filepath)
        print(f"\nAll models saved to: {output_dir}")
    
    def load_all(self, input_dir: str) -> Dict[str, IoTClassifier]:
        """
        Load all models from directory.
        
        Args:
            input_dir: Input directory
            
        Returns:
            Dictionary of loaded models
        """
        import glob
        files = glob.glob(os.path.join(input_dir, "*.joblib"))
        
        for filepath in files:
            model_name = os.path.splitext(os.path.basename(filepath))[0]
            self.trained_models[model_name] = IoTClassifier.load(filepath)
        
        return self.trained_models


def get_available_models() -> Dict[str, str]:
    """
    Get list of available models with descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {name: config['description'] 
            for name, config in IoTClassifier.AVAILABLE_MODELS.items()}


def print_available_models():
    """Print all available models."""
    print("\nAvailable Models:")
    print("-" * 40)
    for name, description in get_available_models().items():
        print(f"  {name}: {description}")
