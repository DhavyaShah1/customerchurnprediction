# CustomerChurnPrediction
ML project to predict telecom customer churn using Logistic Regression and XGBoost models . 

# Problem
Telecom companies lose revenue when customers leave. This project builds a 
model to predict which customers are likely to churn so the business can 
intervene early.

# Dataset
IBM Telco Customer Churn dataset — 7,043 customers, 21 features including 
contract type, monthly charges, tenure, and services used.

# What I Did
- Cleaned data (fixed TotalCharges dtype, handled missing values)
- Handled class imbalance using class_weight='balanced'
- Compared Logistic Regression vs XGBoost
- Evaluated using Recall and ROC-AUC (not just accuracy)

# Key Finding
Balanced Logistic Regression outperformed XGBoost for this use case —
achieving 0.861 ROC-AUC and 84% churn recall.

# Top Churn Drivers
- High monthly charges
- Low tenure (new customers)
- Month-to-month contracts
- Lack of tech support / security services

# Tools Used
Python, Pandas, Scikit-learn, XGBoost, Matplotlib
