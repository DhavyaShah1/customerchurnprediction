# Customer Churn Prediction

## Overview

This project predicts customer churn using the IBM Telco Customer Churn dataset. The objective was to identify customers likely to leave and generate insights that can support retention strategies.

## Business Problem

Customer churn directly impacts recurring revenue. Retaining existing customers is often more cost-effective than acquiring new ones. This project focuses on predicting churn risk early so companies can intervene proactively.

## Dataset

IBM Telco Customer Churn Dataset

Includes:

* Customer demographics
* Contract details
* Billing information
* Services subscribed
* Churn label (Yes / No)

## Project Workflow

1. Data Cleaning

   * Converted `TotalCharges` to numeric
   * Handled missing values
   * Removed customer ID

2. Feature Engineering

   * One-hot encoded categorical variables
   * Train/test split

3. Modeling

   * Logistic Regression
   * Balanced Logistic Regression
   * XGBoost (with tuning)

4. Evaluation Metrics

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * ROC-AUC

## Final Results

| Model                          | Accuracy | Recall (Churn) | F1 Score | ROC-AUC |
| ------------------------------ | -------- | -------------- | -------- | ------- |
| Logistic Regression (Balanced) | 0.76     | 0.84           | 0.65     | 0.861   |
| Tuned XGBoost                  | 0.75     | 0.85           | 0.64     | 0.783   |

## Key Findings

* Customers with higher monthly charges showed higher churn risk
* Customers with shorter tenure were more likely to churn
* Long-term contracts were associated with retention
* Simpler interpretable models outperformed more complex models on this dataset

## Business Recommendations

* Target new customers during first 3 months
* Offer pricing bundles for high monthly charge customers
* Incentivize annual contracts
* Use churn probability scores for outreach prioritization

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib

## Files

* `customerchurn.ipynb` – Full notebook
* `README.md` – Project summary

## Future Improvements

* Hyperparameter optimization with GridSearchCV
* SHAP explainability for XGBoost
* Deployment as Streamlit dashboard
* Automated retention recommendation engine
