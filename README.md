# IoT Device Classification using Machine Learning

This project implements machine learning models for classifying IoT devices based on network traffic data using the UNSW HomeNet dataset.

## Project Overview

The Internet of Things (IoT) refers to networks of connected devices embedded with sensors that collect and exchange data. This project addresses the challenge of IoT device classification, which enables easier network management and security monitoring.

### Objectives
- Research IoT device types and ML applications in IoT classification
- Analyze the UNSW HomeNet IoT dataset
- Implement multiple machine learning models for device classification
- Evaluate and compare model performance
- Identify challenges and future work directions

## Dataset

**UNSW HomeNet Dataset** - A comprehensive IoT network traffic dataset containing:
- Network traffic flows from 105+ IoT and non-IoT devices
- Over 200 million data points
- Features include traffic statistics, protocol features, and temporal characteristics

### Download Links
1. **Kaggle (Recommended)**: https://www.kaggle.com/datasets/mizanunswcyber/iot-and-non-iot-device-classification-dataset
2. **Official UNSW Portal**: https://iotanalytics.unsw.edu.au/homedataset/

## Project Structure

```
IoT-device-classif/
├── data/
│   ├── raw/              # Place downloaded CSV files here
│   └── processed/        # Preprocessed data (auto-generated)
├── models/               # Trained models (auto-generated)
├── results/
│   └── figures/          # Visualization outputs
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading & preprocessing
│   ├── models.py               # ML model definitions
│   └── evaluation.py           # Metrics & visualization
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download and Place Data

1. Download the dataset from Kaggle or the official UNSW portal
2. Extract the CSV files
3. **Place all CSV files in the `data/raw/` directory**

```bash
# Your data structure should look like:
data/
└── raw/
    ├── home_1.csv
    ├── home_2.csv
    └── ... (other CSV files)
```

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Train multiple ML models
3. Evaluate and compare models
4. Generate visualizations and reports

### Run Individual Steps

```bash
# Only preprocess data
python main.py --preprocess-only

# Only train models (requires preprocessed data)
python main.py --train-only

# Only evaluate models (requires trained models)
python main.py --evaluate-only

# Specify custom data path
python main.py --data-path /path/to/your/data
```

## Implemented Models

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of decision trees |
| Decision Tree | Single decision tree classifier |
| XGBoost | Gradient boosting with XGBoost |
| LightGBM | Light Gradient Boosting Machine |
| KNN | K-Nearest Neighbors |
| SVM | Support Vector Machine |
| Logistic Regression | Linear classification |
| MLP | Multi-Layer Perceptron neural network |
| Gradient Boosting | Scikit-learn Gradient Boosting |

## Output

After running the pipeline, you'll find:

### Results Directory (`results/`)
- `evaluation_report.txt` - Detailed evaluation report
- `model_comparison.csv` - Model performance comparison table

### Figures Directory (`results/figures/`)
- `model_comparison.png` - Bar chart comparing all models
- `training_summary.png` - Comprehensive summary visualization
- `confusion_matrix_*.png` - Confusion matrices for each model
- `feature_importance_*.png` - Feature importance plots
- `class_distribution.png` - Data distribution visualization

### Models Directory (`models/`)
- Saved trained models (`.joblib` files)

## Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- Per-class metrics for detailed analysis

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- matplotlib, seaborn
- joblib, tqdm

## Challenges & Future Work

### Challenges
- Class imbalance in device types
- Feature selection for optimal performance
- Scalability with large datasets
- Real-time classification requirements

### Future Work
- Deep learning models (CNN, LSTM)
- Real-time classification system
- Feature engineering improvements
- Cross-network generalization
- Integration with Splunk MLTK

## References

- UNSW IoT Analytics: https://iotanalytics.unsw.edu.au/
- Kaggle Dataset: https://www.kaggle.com/datasets/mizanunswcyber/iot-and-non-iot-device-classification-dataset

## License

This project is for educational purposes.
