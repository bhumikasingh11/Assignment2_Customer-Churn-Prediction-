# Customer Churn Prediction using Logistic Regression

## Overview
This project builds a Logistic Regression model to predict whether a
telecom customer is likely to churn, based on their demographic profile
and service usage patterns. It was developed as part of AI-ML Assignment 2.

## Objective
To analyze customer data and develop a predictive model that identifies
customers at risk of churning, helping the business take proactive
retention measures.

## Dataset
**Telco Customer Churn Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> The dataset is not included in this repository due to licensing.
> Download it from the Kaggle link above and place the CSV file in the
> project directory before running the notebook.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization
- `scikit-learn` — preprocessing, model building, and evaluation

## Methodology
1. **Data Understanding** — Loaded the dataset and identified numerical
   features, categorical features, and the target variable (`Churn`).
2. **Data Preprocessing** — Handled missing values in `TotalCharges`,
   dropped the non-informative `customerID` column, and encoded
   categorical variables using Label Encoding.
3. **Train-Test Split** — Split the data into 80% training and 20%
   testing sets, with feature scaling applied using `StandardScaler`.
4. **Model Development** — Trained a Logistic Regression model on the
   processed training data.
5. **Model Evaluation** — Evaluated predictions on the test set using
   Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.7991 |
| Precision | 0.6426 |
| Recall    | 0.5481 |
| F1-Score  | 0.5916 |

### Observations
1. The model achieves a solid overall accuracy (~80%), indicating it
   correctly classifies most customers.
2. Recall (0.5481) is noticeably lower than precision, meaning the model
   misses a meaningful portion of actual churners — likely a result of
   class imbalance in the dataset.
3. Contract type, tenure, and monthly charges appear to be the strongest
   predictors of churn, consistent with typical telecom churn behavior.

## Conclusion
This project applied Logistic Regression to predict customer churn using
the Telco Customer Churn dataset. After preprocessing — handling missing
values, encoding categorical variables, and scaling numerical features —
the model was trained and evaluated on an 80/20 train-test split,
achieving 79.91% accuracy, 64.26% precision, 54.81% recall, and a 59.16%
F1-score. Contract type, tenure, and monthly charges emerged as key
factors influencing churn, with month-to-month customers and those with
higher charges being more likely to leave. A key limitation of Logistic
Regression is its assumption of a linear relationship between features
and the log-odds of churn, which may not fully capture complex,
non-linear interactions among customer attributes — ensemble methods
like Random Forest or Gradient Boosting could potentially improve
performance, particularly recall on the minority churn class.

