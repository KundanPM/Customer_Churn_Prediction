import joblib
import pandas as pd

model = joblib.load("model/logistic_regression_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

sample = pd.DataFrame({
    "customerID": ["9999-TEST"],
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [12],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [85.5],
    "TotalCharges": [1026.0]
})

sample_processed = preprocessor.transform(sample)

prediction = model.predict(sample_processed)

if prediction[0] == 1:
    print("Prediction: Customer is likely to CHURN.")
else:
    print("Prediction: Customer is NOT likely to CHURN.")

probability = model.predict_proba(sample_processed)

print(f"Stay Probability : {probability[0][0]*100:.2f}%")
print(f"Churn Probability: {probability[0][1]*100:.2f}%")