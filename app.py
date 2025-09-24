from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
model = joblib.load("models/XGBoost_model.pkl")
scaler = joblib.load("models/scaler.pkl")
input_cols = joblib.load("models/columns.pkl")
threshold = joblib.load("models/threshold.pkl")

def log_input(val): 
    return np.log1p(val)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    
    if request.method == "POST":
        try:
            # Get form inputs
            amount = float(request.form["amount"])
            oldbalanceOrg = float(request.form["oldbalanceOrg"])
            newbalanceOrig = float(request.form["newbalanceOrig"])
            oldbalanceDest = float(request.form["oldbalanceDest"])
            newbalanceDest = float(request.form["newbalanceDest"])
            step = int(request.form["step"])
            type_input = request.form["type_input"]

            # Prepare input features
            data = {
                'step': step,
                'log_amount': log_input(amount),
                'log_oldbalanceOrg': log_input(oldbalanceOrg),
                'log_newbalanceOrig': log_input(newbalanceOrig),
                'log_oldbalanceDest': log_input(oldbalanceDest),
                'log_newbalanceDest': log_input(newbalanceDest),
                'balance_diff_sender': oldbalanceOrg - newbalanceOrig,
                'balance_diff_receiver': newbalanceDest - oldbalanceDest,
                'money_moved_ratio': amount / (oldbalanceOrg + 1),
                'type_CASH_IN': int(type_input == 'CASH_IN'),
                'type_CASH_OUT': int(type_input == 'CASH_OUT'),
                'type_DEBIT': int(type_input == 'DEBIT'),
                'type_PAYMENT': int(type_input == 'PAYMENT'),
                'type_TRANSFER': int(type_input == 'TRANSFER'),
                'isFlaggedFraud': 0
            }

            input_df = pd.DataFrame([data])
            input_df = input_df.reindex(columns=input_cols, fill_value=0)

            # Scale numeric cols
            scale_cols = [
                'step',
                'log_amount', 'log_oldbalanceOrg', 'log_newbalanceOrig',
                'log_oldbalanceDest', 'log_newbalanceDest',
                'balance_diff_sender', 'balance_diff_receiver', 'money_moved_ratio'
            ]
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])

            # Predict probability
            proba = model.predict_proba(input_df)[0][1]
            probability = f"{proba:.2%}"

            # Classification
            if proba >= threshold:
                result = f"❗ High-Risk Transaction — Likely Fraud"
            elif proba >= 0.85:
                result = f"⚠️ Suspicious — Manual Review Recommended"
            else:
                result = f"✅ Transaction Likely Safe"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
