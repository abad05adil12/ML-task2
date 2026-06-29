import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

print("Generating 1,500+ realistic customer records...")
np.random.seed(42)
num_records = 1600

customer_ids = [f"CUST-{i:04d}" for i in range(1, num_records + 1)]
ages = np.random.randint(18, 70, size=num_records)
genders = np.random.choice(["Male", "Female"], size=num_records)
cities = np.random.choice(["Sialkot", "Lahore", "Karachi", "Islamabad"], size=num_records)
subscription_types = np.random.choice(["Basic", "Standard", "Premium"], size=num_records)
monthly_spending = np.round(np.random.uniform(10, 150, size=num_records), 2)
tenure = np.random.randint(1, 60, size=num_records)  
num_purchases = np.random.randint(1, 50, size=num_records)
login_frequency = np.random.randint(1, 30, size=num_records)  
satisfaction_score = np.random.randint(1, 6, size=num_records) 

customer_support_requests = np.random.randint(0, 10, size=num_records)
base_prob = 0.1 + (customer_support_requests * 0.08) - (tenure * 0.005) - (satisfaction_score * 0.05)
base_prob = np.clip(base_prob, 0, i1 := 1)
churn_status = np.where(np.random.rand(num_records) < base_prob, 1, 0)

base_date = datetime(2026, 6, 29)
last_activity_dates = [
    (base_date - timedelta(days=int(np.random.randint(0, 90) if status == 1 else np.random.randint(0, 15)))).strftime('%Y-%m-%d')
    for status in churn_status
]

df = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Age': ages,
    'Gender': genders,
    'City': cities,
    'Subscription_Type': subscription_types,
    'Monthly_Spending': monthly_spending,
    'Tenure': tenure,
    'Number_of_Purchases': num_purchases,
    'Customer_Support_Requests': customer_support_requests,
    'Login_Frequency': login_frequency,
    'Last_Activity_Date': last_activity_dates,
    'Satisfaction_Score': satisfaction_score,
    'Churn_Status': churn_status
})

df.to_csv("customer_churn_dataset.csv", index=False)
print("Dataset saved successfully as 'customer_churn_dataset.csv'.\n")

print("Processing data and handling preprocessing...")

df['Last_Activity_Date'] = pd.to_datetime(df['Last_Activity_Date'])
df['Days_Since_Last_Activity'] = (datetime(2026, 6, 29) - df['Last_Activity_Date']).dt.days

X_raw = df.drop(columns=['Customer_ID', 'Last_Activity_Date', 'Churn_Status'])
y = df['Churn_Status']

label_encoders = {}
for col in ['Gender', 'City', 'Subscription_Type']:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col])
    label_encoders[col] = le

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
print(" Preprocessing completed (Encoding & Scaling complete).\n")

print(f"Total Dataset Shape: {df.shape}")
print(f"Churn Distribution:\n{df['Churn_Status'].value_counts(normalize=True) * 100}")
print(f"Average Support Requests for Churned: {df[df['Churn_Status'] == 1]['Customer_Support_Requests'].mean():.2f}")
print(f"Average Support Requests for Retained: {df[df['Churn_Status'] == 0]['Customer_Support_Requests'].mean():.2f}\n")

models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

best_model = None
best_score = 0

print("--- EVALUATION METRICS ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    if auc > best_score:
        best_score = auc
        best_model = model
        best_model_name = name

print(f"\nSelected {best_model_name} as the production deployment model.\n")
def predict_interactive_customer(mock_input):
    input_df = pd.DataFrame([mock_input])
    
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
        
    input_df = input_df[X_raw.columns]
        
    scaled_input = scaler.transform(input_df)
    prob = best_model.predict_proba(scaled_input)[0][1]
    prediction = best_model.predict(scaled_input)[0]
    
    print(f"Raw Input Customer Attributes: {mock_input}")
    print(f"Calculated Churn Probability: {prob * 100:.2f}%")
    print(f"Status Output: {'⚠️ AT RISK (CHURN)' if prediction == 1 else '✅ SAFE (RETAINED)'}")
    print("-" * 50)

high_risk_sample = {
    'Age': 34, 'Gender': 'Female', 'City': 'Sialkot', 'Subscription_Type': 'Basic',
    'Monthly_Spending': 120.0, 'Tenure': 2, 'Number_of_Purchases': 1,
    'Customer_Support_Requests': 8, 'Login_Frequency': 2, 'Days_Since_Last_Activity': 45,
    'Satisfaction_Score': 1
}
predict_interactive_customer(high_risk_sample)