# 🔐 Fraud Detection WebApp

An interactive **Flask** web application that detects potential fraudulent financial transactions using a trained **XGBoost model**. This intelligent tool helps users classify transactions as safe, suspicious, or fraudulent based on various financial indicators.

---

## 📄 Short Description

This project utilizes advanced machine learning techniques to detect fraudulent transactions from financial datasets. By preprocessing transaction data and applying a threshold-optimized **XGBoost classifier**, the app gives users real-time insight into the risk level of any transaction.

---

## 🛠️ Tech Stack

- 🐍 **Python 3.12** – Core language for development  
- ⚡ **XGBoost Classifier** – For high-performance fraud detection  
- 🧪 **Pandas, NumPy** – Data processing and transformation  
- 🌐 **Flask** – For creating a responsive web interface  
- 🧠 **Joblib** – Model and scaler serialization  
- 🔢 **StandardScaler** – For feature normalization  
- 💻 **Bootstrap 5** – For responsive form and layout  

---

## 📊 Data Source

- **Source:** Financial transactions dataset with fraud labels  
- **Features Used:**
  - Step, Type (CASH_OUT, TRANSFER, etc.)
  - Amount, Sender & Receiver Balances (before and after)
  - Engineered features: `log_amount`, `balance_diff_sender`, `money_moved_ratio`, etc.

---

## 🚀 Features & Highlights

### 📌 Problem Statement

Fraud in digital transactions causes significant financial loss globally. This tool empowers institutions or individuals to:

- Assess the risk level of any transaction  
- Take preemptive action against fraud  
- Analyze key patterns in high-risk transactions  

---

## 🎯 Goal of the WebApp

- ✅ Predict fraud risk based on transaction metadata  
- 🧠 Display intelligent results using a tuned model threshold  
- ⚠️ Warn users of suspicious or high-risk financial activity  

---

## 🔍 Walkthrough of Functionality

| Screenshot |
|------------|
| ![WebApp Screenshot](https://github.com/utkarsh2035/Fraud-Detection-WebApp/raw/main/Fraud%20Detection%20System.png) |

- 🧾 **Form Input** — Enter transaction step, amount, balances, and type  
- ⚙️ **XGBoost Classifier** — Provides fraud probability  
- 🎯 **Threshold Logic** — Classifies transaction as:
  - ✅ Safe (low probability)
  - ⚠️ Suspicious (medium probability)
  - ❗ Fraudulent (high probability)
- 📈 **Precision-tuned Threshold** — Achieved F1 score of **0.93** with **Precision > 93%**

---

## 📈 Model Performance

- ✅ XGBoost achieved strong performance on imbalanced data  
- 🎯 F1-optimized threshold with precision-focused validation  
- 🔍 Best threshold: `0.995` (F1-maximized)  
- ❌ No lower threshold achieved precision ≥ 90%  

---

## 📁 Project Structure

Fraud-Detection-WebApp/
│
├── app.py # Flask UI logic
├── models/
│   ├── XGBoost_model.pkl # Trained XGBoost model
│   ├── scaler.pkl # StandardScaler instance
│   ├── columns.pkl # List of features used during training
│   └── threshold.pkl # Threshold for classification
├── templates/
│   └── index.html # Flask HTML template
├── requirements.txt # Required Python packages
├── .gitignore # Ignored files
└── README.md # Project documentation

---

## 💬 Feedback & Improvements

Planned enhancements:

- 🧠 Introduce ensemble voting with multiple models  
- 📊 Display SHAP-based feature importance in UI  
- 🔄 Enable bulk transaction testing via CSV upload  

---

## 🔖 Tags

#Flask #FraudDetection #XGBoost #MachineLearning #DataScience #FinTech #Python #MLProjects #PrecisionFirst #Security
