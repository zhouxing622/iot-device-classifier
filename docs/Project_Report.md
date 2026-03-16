# IoT Device Classification using Machine Learning

## Project Report

**Author:** Josie Zhou  
**Date:** March 2026  
**Platform:** Python, Streamlit  
**Dataset:** UNSW HomeNet

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Dataset Description](#4-dataset-description)
5. [Methodology](#5-methodology)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Model Implementation](#7-model-implementation)
8. [Experimental Results](#8-experimental-results)
9. [Model Comparison and Analysis](#9-model-comparison-and-analysis)
10. [Web Demo Application](#10-web-demo-application)
11. [Challenges and Limitations](#11-challenges-and-limitations)
12. [Future Work](#12-future-work)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Executive Summary

This project implements a machine learning-based IoT device classification system using the UNSW HomeNet dataset. The system analyzes network traffic patterns to identify and classify 14 different types of IoT devices, including smart cameras, audio devices, routers, and various smart home appliances.

**Key Achievements:**
- Trained and evaluated 5 machine learning models
- Achieved **88.78% accuracy** with the XGBoost classifier
- Developed a web-based demo for real-time PCAP file analysis
- Deployed the application to Streamlit Cloud for public access

**Live Demo:** https://iot-device-classifier-josie.streamlit.app/

---

## 2. Introduction

### 2.1 Background

The Internet of Things (IoT) refers to the interconnected network of physical devices embedded with sensors, software, and connectivity capabilities that enable them to collect and exchange data. The proliferation of IoT devices in smart homes, healthcare, industrial settings, and urban infrastructure has created unprecedented opportunities for automation and data-driven decision making.

However, this rapid growth also presents significant challenges:
- **Security vulnerabilities**: Many IoT devices lack robust security measures
- **Network management complexity**: Diverse device types require different handling
- **Resource allocation**: Bandwidth and priority management across heterogeneous devices

### 2.2 Problem Statement

Automatic IoT device classification enables:
1. Enhanced network security through device-specific policies
2. Improved network management and quality of service (QoS)
3. Anomaly detection by understanding normal device behavior
4. Compliance monitoring in enterprise environments

### 2.3 Objectives

1. Research IoT device types and their network traffic characteristics
2. Explore machine learning applications in IoT device classification
3. Implement and compare multiple classification algorithms
4. Develop a practical web-based demonstration application
5. Identify challenges and propose future improvements

---

## 3. Literature Review

### 3.1 IoT Device Classification Approaches

Previous research has explored various approaches to IoT device classification:

| Approach | Description | Advantages | Limitations |
|----------|-------------|------------|-------------|
| **Rule-based** | Manual fingerprinting using known signatures | High precision for known devices | Cannot handle new devices |
| **Statistical** | Analysis of traffic statistics | Simple implementation | Limited accuracy |
| **Machine Learning** | Automated pattern learning | Adaptable, high accuracy | Requires labeled data |
| **Deep Learning** | Neural network-based classification | Can learn complex patterns | Computationally expensive |

### 3.2 Network Traffic Features

Key features used in IoT classification include:
- **Flow-level features**: Duration, packet counts, byte counts
- **Statistical features**: Mean, standard deviation of packet sizes
- **Temporal features**: Inter-arrival times, flow patterns
- **Protocol features**: Port numbers, protocol types, flags

### 3.3 Related Datasets

| Dataset | Devices | Samples | Year |
|---------|---------|---------|------|
| UNSW HomeNet | 105+ | 200M+ | 2024 |
| IoT-23 | 23 | 20M+ | 2020 |
| N-BaIoT | 9 | 7M+ | 2018 |

---

## 4. Dataset Description

### 4.1 UNSW HomeNet Dataset

The UNSW HomeNet dataset is one of the largest publicly available IoT network traffic datasets, collected from real smart home environments.

**Dataset Characteristics:**
- **Source:** University of New South Wales (UNSW) IoT Analytics Lab
- **Collection Period:** Multiple real-world smart home deployments
- **Total Samples:** 933,833 network flows (after preprocessing)
- **Features:** 88 network flow features
- **Device Types:** 14 categories, 25+ specific devices

### 4.2 Device Type Distribution

| Device Type | Sample Count | Percentage |
|-------------|--------------|------------|
| router | 304,262 | 32.6% |
| Camera | 165,752 | 17.7% |
| Audio | 135,957 | 14.6% |
| PC | 132,253 | 14.2% |
| power_switch | 69,912 | 7.5% |
| smartphone | 46,651 | 5.0% |
| Motion_Sensor | 25,462 | 2.7% |
| baby_monitor | 13,135 | 1.4% |
| printer | 11,645 | 1.2% |
| Hub | 9,216 | 1.0% |
| sleep_sensor | 7,703 | 0.8% |
| PowerOutlet | 6,868 | 0.7% |
| Lighting | 4,962 | 0.5% |
| Scale | 55 | 0.01% |

### 4.3 Feature Categories

The dataset includes 75 numeric features after preprocessing:

1. **Basic Flow Features** (4 features)
   - Source/Destination ports, Protocol, Flow Duration

2. **Packet Statistics** (20 features)
   - Forward/Backward packet counts and sizes
   - Min, Max, Mean, Std of packet lengths

3. **Flow Rate Features** (4 features)
   - Bytes/second, Packets/second

4. **Inter-Arrival Time (IAT) Features** (20 features)
   - Flow, Forward, Backward IAT statistics

5. **TCP Flag Features** (10 features)
   - FIN, SYN, RST, PSH, ACK, URG, CWE, ECE counts

6. **Subflow Features** (8 features)
   - Subflow packet and byte statistics

7. **Active/Idle Features** (8 features)
   - Connection activity patterns

---

## 5. Methodology

### 5.1 Overall Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Raw Data   │───▶│ Preprocessing│───▶│  Training   │───▶│ Evaluation │
│  (CSV)      │    │  & Cleaning  │    │   Models    │    │  & Report  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
                          │                   │                  │
                          ▼                   ▼                  ▼
                   ┌──────────────┐    ┌─────────────┐    ┌────────────┐
                   │   Feature    │    │   Model     │    │   Web      │
                   │  Selection   │    │   Saving    │    │   Demo     │
                   └──────────────┘    └─────────────┘    └────────────┘
```

### 5.2 Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.14 |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Visualization | matplotlib, seaborn, plotly |
| Web Framework | Streamlit |
| Deployment | Streamlit Cloud |

---

## 6. Data Preprocessing

### 6.1 Data Loading

The raw dataset was loaded from the UNSW_IoT_Traces.csv file containing network flow records with 89 columns.

```python
# Load dataset
df = pd.read_csv('data/raw/UNSW_IoT_Traces.csv')
# Initial shape: (933,833 rows × 89 columns)
```

### 6.2 Data Cleaning

**Steps performed:**
1. **Duplicate Removal:** 0 duplicate rows found
2. **Missing Value Handling:** Rows with missing target values removed
3. **Infinite Value Replacement:** Replaced inf/-inf with 0

**Final dataset size:** 933,833 samples

### 6.3 Feature Selection

**Excluded columns (non-predictive):**
- Identifiers: `Unnamed: 0`, `FlowID`
- Network addresses: `SrcIP`, `DstIP`, `MAC`
- Timestamps: `Timestamp`
- Target leakage: `DeviceName`, `Source`, `connection_type`

**Selected features:** 75 numeric features

### 6.4 Label Encoding

The target variable `Type` was encoded using LabelEncoder:

| Label | Device Type | Encoding |
|-------|-------------|----------|
| 0 | Audio | 0 |
| 1 | Camera | 1 |
| 2 | Hub | 2 |
| 3 | Lighting | 3 |
| 4 | Motion_Sensor | 4 |
| 5 | PC | 5 |
| 6 | PowerOutlet | 6 |
| 7 | Scale | 7 |
| 8 | baby_monitor | 8 |
| 9 | power_switch | 9 |
| 10 | printer | 10 |
| 11 | router | 11 |
| 12 | sleep_sensor | 12 |
| 13 | smartphone | 13 |

### 6.5 Feature Scaling

StandardScaler was applied to normalize all features:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 6.6 Data Splitting

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 653,682 | 70% |
| Validation | 93,384 | 10% |
| Test | 186,767 | 20% |

Stratified splitting was used to maintain class distribution across all splits.

---

## 7. Model Implementation

### 7.1 Models Selected

Five machine learning algorithms were implemented and compared:

#### 7.1.1 Random Forest

An ensemble method using multiple decision trees with bagging.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Characteristics:**
- Handles non-linear relationships
- Provides feature importance
- Robust to overfitting

#### 7.1.2 Decision Tree

A single tree-based classifier for baseline comparison.

```python
DecisionTreeClassifier(
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

**Characteristics:**
- Interpretable decisions
- Fast training and inference
- Prone to overfitting without pruning

#### 7.1.3 XGBoost

Gradient boosting with regularization for improved performance.

```python
XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
```

**Characteristics:**
- State-of-the-art performance
- Built-in regularization
- Handles missing values

#### 7.1.4 LightGBM

Light Gradient Boosting Machine for efficient training.

```python
LGBMClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
```

**Characteristics:**
- Faster training than XGBoost
- Memory efficient
- Good for large datasets

#### 7.1.5 K-Nearest Neighbors (KNN)

Distance-based classification algorithm.

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    n_jobs=-1
)
```

**Characteristics:**
- Simple and intuitive
- No training phase
- Computationally expensive at inference

### 7.2 Training Process

```python
# Training loop
for model_name in models_to_train:
    classifier = IoTClassifier(model_name)
    classifier.fit(X_train, y_train, X_val, y_val)
    classifier.save(f'models/{model_name}.joblib')
```

**Training Times:**

| Model | Training Time |
|-------|---------------|
| Random Forest | 48.82s |
| Decision Tree | 2.39s |
| XGBoost | 41.32s |
| LightGBM | 43.88s |
| KNN | 0.17s |

---

## 8. Experimental Results

### 8.1 Overall Performance Metrics

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) |
|-------|----------|-------------------|----------------|------------|---------------|
| **XGBoost** | **0.8878** | **0.8443** | **0.7806** | **0.7828** | **0.8769** |
| Decision Tree | 0.8836 | 0.8290 | 0.7587 | 0.7749 | 0.8751 |
| Random Forest | 0.8841 | 0.8519 | 0.7160 | 0.7430 | 0.8717 |
| KNN | 0.8641 | 0.7832 | 0.7573 | 0.7672 | 0.8591 |
| LightGBM | 0.8126 | 0.6073 | 0.6145 | 0.5962 | 0.8104 |

### 8.2 Best Model: XGBoost Detailed Results

**Per-Class Performance:**

| Device Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Audio | 0.98 | 0.97 | 0.98 | 27,191 |
| Camera | 0.95 | 0.80 | 0.87 | 33,151 |
| Hub | 0.71 | 0.84 | 0.77 | 1,843 |
| Lighting | 0.61 | 0.18 | 0.28 | 992 |
| Motion_Sensor | 0.65 | 0.04 | 0.08 | 5,092 |
| PC | 0.98 | 0.99 | 0.99 | 26,451 |
| PowerOutlet | 0.97 | 0.98 | 0.97 | 1,374 |
| Scale | 1.00 | 1.00 | 1.00 | 11 |
| baby_monitor | 0.75 | 0.90 | 0.82 | 2,627 |
| power_switch | 0.87 | 0.93 | 0.90 | 13,982 |
| printer | 0.94 | 0.84 | 0.89 | 2,329 |
| router | 0.80 | 0.93 | 0.86 | 60,853 |
| sleep_sensor | 0.65 | 0.58 | 0.61 | 1,541 |
| smartphone | 0.96 | 0.95 | 0.95 | 9,330 |

### 8.3 Confusion Matrix Analysis

The confusion matrix for XGBoost reveals:

**High Classification Accuracy:**
- PC, Audio, PowerOutlet: >97% accuracy
- smartphone, power_switch: >90% accuracy

**Classification Challenges:**
- Motion_Sensor: Often confused with other sensor types
- Lighting: Limited samples lead to poor recall
- Camera vs Hub: Some misclassification due to similar traffic patterns

### 8.4 Feature Importance Analysis

**Top 10 Most Important Features (XGBoost):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | DstPort | 0.1523 |
| 2 | Protocol | 0.0892 |
| 3 | BwdPktLenMax | 0.0634 |
| 4 | FlowDuration | 0.0521 |
| 5 | TotBwdPkts | 0.0487 |
| 6 | FwdPktLenMean | 0.0445 |
| 7 | BwdPktLenMean | 0.0412 |
| 8 | FlowIATMean | 0.0398 |
| 9 | SrcPort | 0.0376 |
| 10 | PktLenMean | 0.0354 |

**Key Observations:**
- Port numbers are strong indicators (devices use specific ports)
- Packet size statistics help distinguish device types
- Flow duration and inter-arrival times capture behavior patterns

---

## 9. Model Comparison and Analysis

### 9.1 Performance Comparison

```
Accuracy Ranking:
1. XGBoost      ████████████████████████████████████████ 88.78%
2. Random Forest████████████████████████████████████████ 88.41%
3. Decision Tree████████████████████████████████████████ 88.36%
4. KNN          ███████████████████████████████████████  86.41%
5. LightGBM     █████████████████████████████████        81.26%
```

### 9.2 Analysis

#### Why XGBoost Performs Best:
1. **Gradient Boosting:** Iteratively corrects errors from previous trees
2. **Regularization:** L1/L2 penalties prevent overfitting
3. **Handling Imbalanced Data:** Better than random approaches for minority classes

#### Why LightGBM Underperforms:
1. **Leaf-wise growth:** May overfit on imbalanced classes
2. **Default parameters:** Not optimized for this specific dataset
3. **Histogram binning:** May lose precision for continuous features

#### Decision Tree vs Random Forest:
- Similar accuracy (~88%)
- Random Forest provides more stable predictions
- Decision Tree is more interpretable

### 9.3 Trade-offs

| Model | Accuracy | Training Speed | Inference Speed | Interpretability |
|-------|----------|----------------|-----------------|------------------|
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Decision Tree | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| KNN | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| LightGBM | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 10. Web Demo Application

### 10.1 Application Overview

A Streamlit-based web application was developed to demonstrate the trained model in a practical setting.

**Features:**
- PCAP file upload and analysis
- Real-time device classification
- Interactive visualization of results
- Downloadable classification reports

**Live Demo:** https://iot-device-classifier-josie.streamlit.app/

### 10.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Upload    │  │   Results   │  │  Device Detail  │ │
│  │    Page     │  │    Page     │  │      Page       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Backend Processing                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │    PCAP     │  │   Feature   │  │     XGBoost     │ │
│  │   Parser    │  │  Extractor  │  │    Classifier   │ │
│  │   (Scapy)   │  │             │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 10.3 PCAP Processing Pipeline

1. **File Upload:** User uploads PCAP/PCAPNG file
2. **Packet Parsing:** Scapy extracts individual packets
3. **Flow Aggregation:** Packets grouped into network flows
4. **Feature Extraction:** 75 features computed per flow
5. **Classification:** XGBoost model predicts device types
6. **Visualization:** Results displayed with confidence scores

### 10.4 User Interface

The application provides:
- **Upload Page:** File selection and analysis configuration
- **Results Page:** Overview of detected devices with statistics
- **Detail Page:** Per-device flow analysis and filtering
- **Export:** CSV download of classification results

---

## 11. Challenges and Limitations

### 11.1 Data Challenges

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Class Imbalance** | Poor performance on minority classes | Stratified sampling, class weights |
| **Scale class (55 samples)** | Unreliable predictions | More data collection needed |
| **Motion_Sensor confusion** | 4% recall | Feature engineering for sensor types |

### 11.2 Technical Limitations

1. **Feature Extraction Accuracy**
   - Custom feature extractor may not perfectly match training data format
   - Some advanced features (Active/Idle times) are approximated

2. **Real-time Performance**
   - Current implementation processes files batch-wise
   - Not suitable for streaming network analysis

3. **Generalization**
   - Model trained on specific smart home environment
   - May not generalize to enterprise or industrial IoT

### 11.3 Model Limitations

- **Logistic Regression:** Too slow for large dataset, excluded from final comparison
- **Deep Learning:** Not implemented due to time constraints
- **Hyperparameter Tuning:** Default parameters used; optimization could improve results

---

## 12. Future Work

### 12.1 Short-term Improvements

1. **Hyperparameter Optimization**
   - Grid search or Bayesian optimization
   - Expected improvement: 2-5% accuracy

2. **Handling Class Imbalance**
   - SMOTE oversampling
   - Class weight adjustment
   - Focal loss for minority classes

3. **Feature Engineering**
   - Domain-specific features for sensor types
   - Time-series features for traffic patterns

### 12.2 Long-term Extensions

1. **Deep Learning Models**
   - CNN for spatial patterns in traffic
   - LSTM for temporal dependencies
   - Transformer-based architectures

2. **Real-time Classification**
   - Stream processing with Apache Kafka
   - Edge deployment for low-latency inference

3. **Incremental Learning**
   - Online learning for new device types
   - Concept drift detection and adaptation

4. **Integration with Network Infrastructure**
   - SDN controller integration
   - Automated policy enforcement
   - Splunk MLTK integration for enterprise deployment

### 12.3 Research Directions

- Zero-shot learning for unknown device types
- Federated learning for privacy-preserving classification
- Adversarial robustness against traffic manipulation

---

## 13. Conclusion

This project successfully implemented an IoT device classification system using machine learning techniques. The key findings and contributions are:

### 13.1 Summary of Results

- **Best Model:** XGBoost with 88.78% accuracy
- **Device Coverage:** 14 IoT device types classified
- **Key Features:** Port numbers, packet statistics, and flow characteristics are most predictive
- **Practical Application:** Web demo deployed for PCAP file analysis

### 13.2 Key Contributions

1. Comprehensive comparison of 5 machine learning algorithms for IoT classification
2. Identification of most important network traffic features
3. Development of an accessible web-based demonstration tool
4. Analysis of challenges and future research directions

### 13.3 Practical Impact

The developed system can be applied to:
- Network security monitoring
- Smart home device management
- Enterprise IoT asset discovery
- Research and educational purposes

### 13.4 Final Remarks

IoT device classification using machine learning is a promising approach for addressing the challenges of heterogeneous IoT networks. While the current system achieves high accuracy for most device types, further improvements in handling class imbalance and rare device types will enhance practical applicability.

---

## 14. References

1. Rahman, M. M., et al. (2025). "UNSW HomeNet: A network traffic flow dataset for AI-based smart home device classification." *Computers & Industrial Engineering*, 204.

2. Sivanathan, A., et al. (2018). "Classifying IoT devices in smart environments using network traffic characteristics." *IEEE Transactions on Mobile Computing*.

3. Meidan, Y., et al. (2018). "N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders." *IEEE Pervasive Computing*.

4. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD*.

5. UNSW IoT Analytics Lab. https://iotanalytics.unsw.edu.au/

6. Kaggle Dataset. https://www.kaggle.com/datasets/mizanunswcyber/iot-and-non-iot-device-classification-dataset

---

## Appendix A: Project Structure

```
IoT-device-classif/
├── data/
│   ├── raw/                    # Raw CSV data
│   └── processed/              # Preprocessed data
│       ├── scaler.joblib
│       ├── label_encoder.joblib
│       └── metadata.joblib
├── demo/
│   ├── app.py                  # Streamlit application
│   └── feature_extractor.py    # PCAP feature extraction
├── models/
│   ├── xgboost.joblib          # Best model
│   ├── decision_tree.joblib
│   └── lightgbm.joblib
├── results/
│   ├── evaluation_report.txt
│   ├── model_comparison.csv
│   └── figures/                # Visualization outputs
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── main.py                     # Main execution script
├── requirements.txt
└── README.md
```

## Appendix B: How to Reproduce

```bash
# Clone repository
git clone https://github.com/zhouxing622/iot-device-classifier.git
cd iot-device-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/

# Run training pipeline
python main.py

# Run web demo
streamlit run demo/app.py
```

---

*End of Report*
