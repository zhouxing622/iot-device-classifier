"""
Data Preprocessing Module for IoT Device Classification

This module handles loading, cleaning, and preprocessing the UNSW HomeNet dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class IoTDataPreprocessor:
    """Preprocessor for UNSW HomeNet IoT dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the raw data directory or CSV file
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Args:
            file_pattern: Pattern for CSV files to load
            
        Returns:
            Combined DataFrame
        """
        if os.path.isfile(self.data_path):
            print(f"Loading single file: {self.data_path}")
            df = pd.read_csv(self.data_path)
        else:
            import glob
            files = glob.glob(os.path.join(self.data_path, file_pattern))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {self.data_path}")
            
            print(f"Found {len(files)} CSV files")
            dfs = []
            for f in files:
                print(f"  Loading: {os.path.basename(f)}")
                dfs.append(pd.read_csv(f))
            df = pd.concat(dfs, ignore_index=True)
        
        print(f"Total samples loaded: {len(df):,}")
        return df
    
    def explore_data(self, df: pd.DataFrame) -> dict:
        """
        Perform initial data exploration.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing exploration results
        """
        exploration = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        }
        
        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)
        print(f"Dataset Shape: {exploration['shape'][0]:,} rows × {exploration['shape'][1]} columns")
        print(f"\nColumn Types:")
        for col, dtype in exploration['dtypes'].items():
            missing = exploration['missing_values'][col]
            missing_pct = exploration['missing_percentage'][col]
            print(f"  {col}: {dtype} (missing: {missing:,} / {missing_pct:.2f}%)")
        
        return exploration
    
    def identify_target_column(self, df: pd.DataFrame) -> str:
        """
        Identify the target column for device classification.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the target column
        """
        potential_targets = ['Type', 'DeviceName', 'device', 'device_type', 'label', 'class', 
                           'device_name', 'Device', 'Label', 'Class',
                           'device_category', 'category']
        
        for col in potential_targets:
            if col in df.columns:
                self.target_column = col
                print(f"\nIdentified target column: '{col}'")
                print(f"Unique classes: {df[col].nunique()}")
                print(f"\nClass distribution:")
                print(df[col].value_counts())
                return col
        
        print("\nAvailable columns:", df.columns.tolist())
        raise ValueError("Could not automatically identify target column. "
                        "Please specify manually.")
    
    def clean_data(self, df: pd.DataFrame, 
                   target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and removing duplicates.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (optional)
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        initial_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {initial_rows - len(df):,} duplicate rows")
        
        if target_col:
            df = df.dropna(subset=[target_col])
            print(f"Removed rows with missing target values")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_col and df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col] = df[col].fillna(mode_val)
        
        print(f"Final dataset size: {len(df):,} rows")
        
        return df
    
    def select_features(self, df: pd.DataFrame, 
                        target_col: str,
                        exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select features for model training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if exclude_cols is None:
            exclude_cols = []
        
        cols_to_exclude = [target_col] + exclude_cols
        
        non_feature_patterns = ['mac', 'ip', 'address', 'timestamp', 'time', 
                                'date', 'id', 'index', 'name', 'flowid', 'source',
                                'connection_type', 'unnamed', 'devicename']
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in non_feature_patterns:
                if pattern in col_lower and col not in cols_to_exclude:
                    cols_to_exclude.append(col)
                    break
        
        feature_cols = [col for col in df.columns if col not in cols_to_exclude]
        
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = numeric_features
        
        print(f"\nSelected {len(numeric_features)} numeric features for training")
        print(f"Excluded columns: {cols_to_exclude}")
        
        X = df[numeric_features]
        y = df[target_col]
        
        return X, y
    
    def encode_labels(self, y: pd.Series) -> Tuple[np.ndarray, dict]:
        """
        Encode target labels to numeric values.
        
        Args:
            y: Target Series
            
        Returns:
            Tuple of (encoded labels, label mapping)
        """
        y_encoded = self.label_encoder.fit_transform(y)
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                 range(len(self.label_encoder.classes_))))
        
        print(f"\nLabel encoding mapping:")
        for label, code in label_mapping.items():
            count = (y == label).sum()
            print(f"  {code}: {label} ({count:,} samples)")
        
        return y_encoded, label_mapping
    
    def scale_features(self, X: pd.DataFrame, 
                       fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
            
        Returns:
            Dictionary containing split data
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def preprocess_pipeline(self, 
                            target_col: Optional[str] = None,
                            test_size: float = 0.2,
                            val_size: float = 0.1) -> dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            target_col: Target column name (auto-detected if None)
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        df = self.load_data()
        
        self.explore_data(df)
        
        if target_col is None:
            target_col = self.identify_target_column(df)
        self.target_column = target_col
        
        df = self.clean_data(df, target_col)
        
        X, y = self.select_features(df, target_col)
        
        y_encoded, label_mapping = self.encode_labels(y)
        
        X_scaled = self.scale_features(X)
        
        data_splits = self.split_data(X_scaled, y_encoded, test_size, val_size)
        
        result = {
            **data_splits,
            'feature_names': self.feature_columns,
            'label_mapping': label_mapping,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'n_classes': len(label_mapping),
            'n_features': len(self.feature_columns)
        }
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Features: {result['n_features']}")
        print(f"Classes: {result['n_classes']}")
        
        return result


def save_processed_data(data: dict, output_dir: str):
    """
    Save processed data to files.
    
    Args:
        data: Dictionary from preprocess_pipeline
        output_dir: Output directory path
    """
    import joblib
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(output_dir, 'X_val.npy'), data['X_val'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(output_dir, 'y_val.npy'), data['y_val'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data['y_test'])
    
    metadata = {
        'feature_names': data['feature_names'],
        'label_mapping': data['label_mapping'],
        'n_classes': data['n_classes'],
        'n_features': data['n_features']
    }
    joblib.dump(metadata, os.path.join(output_dir, 'metadata.joblib'))
    joblib.dump(data['label_encoder'], os.path.join(output_dir, 'label_encoder.joblib'))
    joblib.dump(data['scaler'], os.path.join(output_dir, 'scaler.joblib'))
    
    print(f"\nProcessed data saved to: {output_dir}")


def load_processed_data(input_dir: str) -> dict:
    """
    Load processed data from files.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        Dictionary containing processed data
    """
    import joblib
    
    data = {
        'X_train': np.load(os.path.join(input_dir, 'X_train.npy')),
        'X_val': np.load(os.path.join(input_dir, 'X_val.npy')),
        'X_test': np.load(os.path.join(input_dir, 'X_test.npy')),
        'y_train': np.load(os.path.join(input_dir, 'y_train.npy')),
        'y_val': np.load(os.path.join(input_dir, 'y_val.npy')),
        'y_test': np.load(os.path.join(input_dir, 'y_test.npy')),
    }
    
    metadata = joblib.load(os.path.join(input_dir, 'metadata.joblib'))
    data.update(metadata)
    
    data['label_encoder'] = joblib.load(os.path.join(input_dir, 'label_encoder.joblib'))
    data['scaler'] = joblib.load(os.path.join(input_dir, 'scaler.joblib'))
    
    print(f"Loaded processed data from: {input_dir}")
    print(f"  Training samples: {len(data['X_train']):,}")
    print(f"  Features: {data['n_features']}")
    print(f"  Classes: {data['n_classes']}")
    
    return data
