# 📊 ForecastIQ — Enterprise Sales Forecasting System

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-blue?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Prophet](https://img.shields.io/badge/Prophet-f37626?style=for-the-badge)](https://facebook.github.io/prophet/)

An end-to-end, production-grade time series forecasting pipeline designed to predict weekly sales across 43 US states. The system leverages a multi-model architecture with automated "champion" model selection to ensure peak predictive accuracy for each regional market.

---

## 🚀 Key Features

*   **Multi-Model Engine:** Supports SARIMA, Facebook Prophet, XGBoost, and LSTM architectures.
*   **Automated Selection:** Dynamically selects the best-performing model per state based on Mean Absolute Error (MAE).
*   **Advanced Feature Engineering:**
    *   Temporal lags (t-1, t-7, t-30) and rolling statistics (mean/std).
    *   Calendar-aware encodings (Week, Month, Holiday flags).
*   **Leakage-Free Validation:** Strict time-based splitting to ensure real-world performance reliability.
*   **Interactive Analytics:** Built-in dashboard for visual trend analysis and model comparison.

## 🛠 Tech Stack

*   **Logic:** Python 3.x
*   **ML Frameworks:** Scikit-learn, XGBoost, Prophet, Statsmodels, TensorFlow/Keras
*   **API:** FastAPI with Uvicorn
*   **Data Processing:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn, Plotly

## 📁 Project Structure

```text
├── api/                # FastAPI server and endpoint definitions
├── charts/             # Generated EDA and performance visualizations
├── data/               # Raw/Cleaned datasets and serialized (.joblink) models
├── notebooks/          # Step-by-step EDA, training, and evaluation workflows
├── src/                # Modular source code for data loading and processing
├── dashboard.html      # Interactive frontend for forecast visualization
├── verify_training.py  # Validation script for model training integrity
└── generate_charts.py  # Automated visualization generation scripts
```

## ⚙️ Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
Navigate to the API directory and launch the server:
```bash
cd api
uvicorn main:app --reload --port 8003
```

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/forecast` | `GET` | Returns an 8-week forecast for a specific state (e.g., `?state=Kansas`). |
| `/summary` | `GET` | Provides a high-level system overview and metrics. |
| `/models` | `GET` | Returns a detailed MAE comparison of all models for a given state. |

## 📈 Performance Summary

| Metric | Target / Result |
| :--- | :--- |
| **Coverage** | 43 US States |
| **Forecast Horizon** | 8 Weeks |
| **Primary Champion** | XGBoost (Lowest global MAE) |
| **Average MAE** | ~97.5M |

---
*Developed as a robust solution for regional sales demand planning and inventory optimization.*
