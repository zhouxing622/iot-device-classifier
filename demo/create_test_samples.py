"""
Create test samples from the original dataset for validation.
This helps verify if the demo is working correctly.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_validation_samples():
    """Create validation samples from the original dataset."""
    
    print("Loading original dataset...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'data', 'raw', 'UNSW_IoT_Traces.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    print("\nSampling 10 flows from each device type...")
    
    samples = []
    for device_type in df['Type'].unique():
        device_samples = df[df['Type'] == device_type].sample(
            n=min(10, len(df[df['Type'] == device_type])),
            random_state=42
        )
        samples.append(device_samples)
    
    validation_df = pd.concat(samples, ignore_index=True)
    
    output_path = os.path.join(os.path.dirname(__file__), 'validation_samples.csv')
    validation_df.to_csv(output_path, index=False)
    
    print(f"\nValidation samples saved to: {output_path}")
    print(f"Total samples: {len(validation_df)}")
    print("\nSamples per device type:")
    print(validation_df['Type'].value_counts())
    
    return validation_df


def validate_model_predictions():
    """Validate model predictions against known labels."""
    
    print("\n" + "="*60)
    print("MODEL VALIDATION")
    print("="*60)
    
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, 'models', 'xgboost.joblib')
    scaler_path = os.path.join(base_path, 'data', 'processed', 'scaler.joblib')
    metadata_path = os.path.join(base_path, 'data', 'processed', 'metadata.joblib')
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = joblib.load(scaler_path)
    metadata = joblib.load(metadata_path)
    
    validation_path = os.path.join(os.path.dirname(__file__), 'validation_samples.csv')
    if not os.path.exists(validation_path):
        print("Creating validation samples first...")
        create_validation_samples()
    
    df = pd.read_csv(validation_path)
    
    feature_cols = metadata['feature_names']
    X = df[feature_cols].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    label_mapping_inv = {v: k for k, v in metadata['label_mapping'].items()}
    predicted_labels = [label_mapping_inv[p] for p in predictions]
    
    true_labels = df['Type'].tolist()
    
    print("\n" + "-"*60)
    print("VALIDATION RESULTS")
    print("-"*60)
    
    correct = 0
    total = len(true_labels)
    
    print(f"\n{'True Label':<20} {'Predicted':<20} {'Confidence':<12} {'Correct'}")
    print("-"*65)
    
    for i, (true, pred, prob) in enumerate(zip(true_labels, predicted_labels, probabilities)):
        confidence = max(prob) * 100
        is_correct = true == pred
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{true:<20} {pred:<20} {confidence:>8.1f}%    {status}")
    
    accuracy = correct / total * 100
    
    print("\n" + "="*60)
    print(f"VALIDATION ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    print("="*60)
    
    if accuracy >= 85:
        print("\n✅ Model is working correctly! Results can be trusted.")
    elif accuracy >= 70:
        print("\n⚠️ Model accuracy is acceptable but not optimal.")
    else:
        print("\n❌ Model accuracy is lower than expected. Check the pipeline.")
    
    return accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate IoT classifier')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create validation samples')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation')
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_validation_samples()
    elif args.validate:
        validate_model_predictions()
    else:
        create_validation_samples()
        validate_model_predictions()
