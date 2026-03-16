"""
IoT Device Classification using Machine Learning

Main execution script for training and evaluating IoT device classification models
using the UNSW HomeNet dataset.

Usage:
    python main.py                    # Run full pipeline
    python main.py --preprocess-only  # Only preprocess data
    python main.py --train-only       # Only train models (requires preprocessed data)
    python main.py --evaluate-only    # Only evaluate (requires trained models)
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')

from src.data_preprocessing import (
    IoTDataPreprocessor, 
    save_processed_data, 
    load_processed_data
)
from src.models import (
    IoTClassifier, 
    ModelTrainer, 
    print_available_models
)
from src.evaluation import (
    ModelEvaluator, 
    Visualizer, 
    generate_report
)


# Configuration
CONFIG = {
    'raw_data_path': 'data/raw/UNSW_IoT_Traces.csv',
    'processed_data_path': 'data/processed',
    'models_path': 'models',
    'results_path': 'results',
    'figures_path': 'results/figures',
    'test_size': 0.2,
    'val_size': 0.1,
    'target_column': 'Type',
    'models_to_train': [
        'random_forest',
        'decision_tree', 
        'xgboost',
        'lightgbm',
        'knn'
    ]
}


def check_data_exists():
    """Check if raw data exists."""
    raw_path = CONFIG['raw_data_path']
    
    if os.path.isfile(raw_path):
        print(f"\nUsing data file: {raw_path}")
        return True
    
    if not os.path.exists(raw_path):
        print(f"\n{'='*60}")
        print("DATA NOT FOUND!")
        print(f"{'='*60}")
        print(f"\nPlease download the UNSW HomeNet dataset and place it in:")
        print(f"  {os.path.abspath(raw_path)}")
        print(f"\nDownload from:")
        print("  1. Kaggle: https://www.kaggle.com/datasets/mizanunswcyber/iot-and-non-iot-device-classification-dataset")
        print("  2. Official: https://iotanalytics.unsw.edu.au/homedataset/")
        return False
    
    import glob
    csv_files = glob.glob(os.path.join(raw_path, '*.csv'))
    
    if not csv_files:
        print(f"\n{'='*60}")
        print("NO CSV FILES FOUND!")
        print(f"{'='*60}")
        print(f"\nThe data directory exists but contains no CSV files.")
        print(f"Please place your CSV data files in:")
        print(f"  {os.path.abspath(raw_path)}/")
        return False
    
    print(f"\nFound {len(csv_files)} CSV file(s) in {raw_path}/")
    for f in csv_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")
    
    return True


def run_preprocessing():
    """Run data preprocessing pipeline."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    if not check_data_exists():
        return None
    
    preprocessor = IoTDataPreprocessor(CONFIG['raw_data_path'])
    
    data = preprocessor.preprocess_pipeline(
        target_col=CONFIG.get('target_column'),
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )
    
    save_processed_data(data, CONFIG['processed_data_path'])
    
    return data


def run_training(data: dict = None):
    """Run model training."""
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    if data is None:
        if not os.path.exists(CONFIG['processed_data_path']):
            print("Processed data not found. Running preprocessing first...")
            data = run_preprocessing()
            if data is None:
                return None, None
        else:
            data = load_processed_data(CONFIG['processed_data_path'])
    
    print_available_models()
    print(f"\nTraining models: {CONFIG['models_to_train']}")
    
    trainer = ModelTrainer(CONFIG['models_to_train'])
    
    trained_models = trainer.train_all(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    trainer.save_all(CONFIG['models_path'])
    
    return trained_models, data


def run_evaluation(trained_models: dict = None, data: dict = None):
    """Run model evaluation."""
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    if data is None:
        data = load_processed_data(CONFIG['processed_data_path'])
    
    if trained_models is None:
        trainer = ModelTrainer(CONFIG['models_to_train'])
        trained_models = trainer.load_all(CONFIG['models_path'])
    
    evaluator = ModelEvaluator(data['label_mapping'], CONFIG['figures_path'])
    visualizer = Visualizer(data['label_mapping'], CONFIG['figures_path'])
    
    results = {}
    
    for model_name, model in trained_models.items():
        y_pred = model.predict(data['X_test'])
        
        metrics = evaluator.evaluate_model(
            data['y_test'], y_pred, model_name
        )
        results[model_name] = metrics
        
        evaluator.print_evaluation(metrics)
        
        visualizer.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            model_name, 
            normalize=True
        )
        
        importance = model.get_feature_importance(data['feature_names'])
        if importance:
            visualizer.plot_feature_importance(importance, model_name, top_n=15)
    
    comparison_df = evaluator.compare_models(results)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    visualizer.plot_model_comparison(comparison_df)
    visualizer.plot_class_distribution(data['y_test'], 'Test Set Class Distribution')
    visualizer.plot_training_summary(results)
    
    report_path = os.path.join(CONFIG['results_path'], 'evaluation_report.txt')
    generate_report(results, comparison_df, report_path)
    
    comparison_csv_path = os.path.join(CONFIG['results_path'], 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparison saved to: {comparison_csv_path}")
    
    return results, comparison_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='IoT Device Classification using Machine Learning'
    )
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run data preprocessing')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run model training')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to raw data (overrides config)')
    
    args = parser.parse_args()
    
    if args.data_path:
        CONFIG['raw_data_path'] = args.data_path
    
    print("\n" + "="*60)
    print("IoT DEVICE CLASSIFICATION USING MACHINE LEARNING")
    print("UNSW HomeNet Dataset")
    print("="*60)
    
    try:
        if args.preprocess_only:
            run_preprocessing()
        elif args.train_only:
            run_training()
        elif args.evaluate_only:
            run_evaluation()
        else:
            data = run_preprocessing()
            if data is None:
                print("\nPreprocessing failed. Please check your data.")
                sys.exit(1)
            
            trained_models, data = run_training(data)
            if trained_models is None:
                print("\nTraining failed.")
                sys.exit(1)
            
            results, comparison_df = run_evaluation(trained_models, data)
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETE!")
            print("="*60)
            print(f"\nResults saved in: {os.path.abspath(CONFIG['results_path'])}/")
            print(f"Models saved in: {os.path.abspath(CONFIG['models_path'])}/")
            print(f"\nBest model: {comparison_df.iloc[0]['Model']}")
            print(f"Best F1-Score: {comparison_df.iloc[0]['F1 (Macro)']:.4f}")
            
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
