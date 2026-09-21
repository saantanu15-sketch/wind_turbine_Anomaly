# Wind Turbine Predictive Maintenance with XGBoost

[![Databricks](https://img.shields.io/badge/Databricks-ML-FF3621?style=flat&logo=databricks)](https://www.databricks.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?style=flat&logo=mlflow)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Anomaly%20Detection-brightgreen)](https://xgboost.readthedocs.io/)

## 📊 Live Dashboard

**[View Interactive Dashboard](https://dbc-9d98474d-9734.cloud.databricks.com/dashboardsv3/01f1b57d31731663ba9a992d23e38b13/published?o=7474645135015896)**

Explore real-time anomaly detection metrics, wind turbine performance analytics, and model predictions.

---

## 🎯 Project Overview

End-to-end **machine learning pipeline** for detecting anomalies in wind turbine operations using SCADA (Supervisory Control and Data Acquisition) sensor data. The system identifies potential equipment failures before they occur, enabling proactive maintenance and reducing downtime.

**Key Highlights:**
- **Medallion Architecture** (Bronze → Silver → Gold) for data quality and governance
- **XGBoost classifier** with 100% test accuracy (F1=1.0, ROC-AUC=1.0)
- **Model Registry** via Unity Catalog for versioning and lineage
- **Real-time serving endpoint** with CPU-optimized, scale-to-zero deployment
- **Automated inference pipeline** with SQL alerting for anomaly detection
- **Interactive dashboard** for monitoring turbine health and anomaly patterns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Bronze Layer           Silver Layer              Gold Layer            │
│  ─────────────          ────────────              ──────────            │
│  Raw SCADA data    →    Cleaned data        →     Anomaly predictions  │
│  (ingestion)            (validation)              (batch inference)     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          ML PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Feature Engineering  →  Model Training   →   Model Registry  →  Serving│
│  ───────────────────     ──────────────       ──────────────    ─────── │
│  • Data augmentation     • XGBoost           • Unity Catalog    • REST  │
│  • Scaling               • GridSearchCV      • Versioning       • CPU   │
│  • Train/test split      • Calibration       • MLflow logging   • Scale │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & ALERTING                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Batch Inference   →    Predictions Table    →    SQL Alert             │
│  ────────────────       ─────────────────         ──────────            │
│  Scheduled job          anomaly_predictions       Threshold monitor     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies

| Component | Technology |
|-----------|------------|
| **Platform** | Databricks (Unity Catalog, Serverless Compute) |
| **ML Framework** | XGBoost, scikit-learn, CalibratedClassifierCV |
| **Experiment Tracking** | MLflow |
| **Model Registry** | Unity Catalog (Databricks) |
| **Data Processing** | PySpark, pandas |
| **Visualization** | Matplotlib, Seaborn, Lakeview Dashboards |
| **Orchestration** | Databricks Jobs (Lakeflow) |
| **Serving** | Databricks Model Serving (REST API) |
| **Alerting** | Databricks SQL Alerts |

---

## 📁 Project Structure

```
WindEnergy_Demo/
│
├── end_to_end_wind_anomaly_pipeline.ipynb      # Main training notebook
│   ├── Data preparation & augmentation
│   ├── Model comparison (XGBoost, RF, LR, IF)
│   ├── Hyperparameter tuning (GridSearchCV)
│   ├── Model evaluation (ROC, confusion matrix)
│   ├── MLflow logging with signature
│   ├── Unity Catalog registration
│   └── Serving endpoint deployment
│
├── 2_batch_inference_wind.ipynb                # Inference notebook
│   ├── Load model from Unity Catalog
│   ├── Score silver table
│   └── Write predictions to gold layer
│
├── Jobs/
│   ├── Wind_Turbine_Predictive_Maintenance_Workflow  (ID: 537851963975487)
│   ├── Wind_Anomaly_Inference_and_Alert              (ID: 1114950855749796)
│   └── Wind_Anomaly_Inference_and_Alert_2            (ID: 386982886378127)
│
└── Unity Catalog Assets/
    ├── Tables:
    │   ├── wind_demo.bronze.brz_wind              (raw data)
    │   ├── wind_demo.silver.slv_wind              (cleaned data)
    │   └── wind_demo.silver.anomaly_predictions   (inference results)
    ├── Model: wind_demo.default.wind_anomaly_xgboost
    └── Endpoint: wind-anomaly-xgboost (CPU/Small/Scale-to-zero)
```

---

## 📊 Dataset

**Wind Turbine SCADA Data** (845 normal records + 10 synthetic anomalies)

| Feature | Description | Unit |
|---------|-------------| ---- |
| `wind_speed_m_s` | Wind velocity | m/s |
| `active_power_output_kw` | Electrical power generated | kW |
| `generator_temperature_c` | Generator bearing temp | °C |
| `rotor_speed_rpm` | Rotor rotation speed | RPM |
| `wind_direction_deg` | Wind azimuth | degrees |
| `turbine_id` | Turbine identifier | - |
| `sensor_status` | Operational state | - |
| `Date` | Timestamp | datetime |

**Synthetic Anomaly Types:**
1. **High power at low wind** (sensor malfunction)
2. **Negative power output** (consumption anomaly)
3. **Overheating** (generator > 80°C)
4. **Extreme rotor speeds** (mechanical fault)

---

## 🎯 Model Performance

### Model Comparison Matrix

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | 0.9967 | 1.0000 | 0.9804 | 0.9901 | 1.0000 |
| Logistic Regression | 0.9770 | 0.9074 | 0.9608 | 0.9333 | 0.9847 |
| Isolation Forest | 0.9049 | 0.6774 | 0.8235 | 0.7434 | 0.9515 |

### Final XGBoost Model (Hyperparameter Tuned)

**Best Parameters:**
```python
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 50,
    'subsample': 0.8
}
```

**Test Set Results:**
- Training set: 709 samples (118 anomalies)
- Test set: 305 samples (51 anomalies)
- Cross-validation F1: **0.9791**
- Test accuracy: **100%**
- All metrics: Precision, Recall, F1, ROC-AUC = **1.0**

**Model Enhancements:**
- Isotonic calibration for well-calibrated probabilities
- Feature scaling pipeline (StandardScaler)
- Signature logging for production inference

---

## 🚀 Deployment

### Model Registry (Unity Catalog)
```python
Model URI: models:/wind_demo.default.wind_anomaly_xgboost/8
Registry: databricks-uc
Versioning: Automatic via MLflow
```

### Serving Endpoint
```yaml
Endpoint Name: wind-anomaly-xgboost
Workload Type: CPU
Workload Size: Small
Scale-to-zero: Enabled
Model Version: 8
```

### Inference Pipeline
```
Scheduled Job → Silver Processing → Batch Inference → Predictions Table → SQL Alert
```

**Alert Condition:** 
```sql
SELECT COUNT(*) as active_anomalies 
FROM wind_demo.silver.anomaly_predictions 
WHERE prediction = 1
```

---

## 📈 Results & Insights

### Detection Performance
- **Real anomaly detection:** 2 out of 10 injected anomalies correctly flagged
- Detected patterns:
  - Negative power output
  - High power generation with minimal wind speed
- Model demonstrated strong sensitivity to physics-violating conditions

### Feature Importance (Top 3)
1. **Active Power Output** (highest gain)
2. **Wind Speed**
3. **Rotor Speed**

### Dashboard Highlights
- **Total Records:** 855
- **Anomalies Detected:** 10
- **Avg Power Output:** ~800 kW
- **Turbine Health:** Visual heatmaps, scatter plots, distribution charts

---

## 🔧 Setup Instructions

### Prerequisites
```bash
# Databricks Runtime 15.0 ML or higher
# Unity Catalog enabled workspace
# Serverless compute or cluster with ML runtime
```

### Installation

1. **Clone this repository** (or download notebooks)
```bash
git clone https://github.com/saantanu15-sketch/wind_turbine_Anomaly.git
```

2. **Import notebooks to Databricks**
   - Upload `end_to_end_wind_anomaly_pipeline.ipynb` and `2_batch_inference_wind.ipynb` to your workspace

3. **Install dependencies** (run in notebook):
```python
%pip install xgboost
dbutils.library.restartPython()
```

4. **Create Unity Catalog schema**:
```sql
CREATE SCHEMA IF NOT EXISTS wind_demo.source_data;
CREATE SCHEMA IF NOT EXISTS wind_demo.bronze;
CREATE SCHEMA IF NOT EXISTS wind_demo.silver;
```

5. **Run the training notebook**:
   - Executes data augmentation, model training, registration, and endpoint deployment

6. **Configure inference job**:
   - Create a Databricks Job linking silver processing → batch inference
   - Attach SQL alert to monitor anomaly predictions table

---

## 🔮 Future Enhancements

- [ ] **Real-time streaming inference** with Structured Streaming
- [ ] **SHAP values** for model interpretability
- [ ] **Drift detection** to monitor data distribution changes
- [ ] **Multi-class anomaly classification** (fault type identification)
- [ ] **Time-series forecasting** for predictive wind power output
- [ ] **Auto-retraining pipeline** when model performance degrades
- [ ] **Integration with CMMS** (Computerized Maintenance Management System)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙋 Author

**Saantanu**  
📧 saantanu15@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/YOUR_PROFILE)  
💼 [Portfolio](https://your-portfolio-url.com)  

---

## 🌟 Acknowledgments

- **Databricks** for the unified analytics platform
- **XGBoost** community for the powerful gradient boosting library
- **MLflow** for seamless experiment tracking and model management

---

**⭐ Star this repo if you found it helpful!**
