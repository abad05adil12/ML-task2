# Technical Analysis Report: Customer Churn Prediction System (ML-2)

## 1. Dataset Description
For this task, a synthetic customer base of 1,600 unique rows was constructed to simulate true engagement records within telecom and SaaS ecosystems. The structural properties tracked are:
- Demographics: Age, Gender, City.
- Engagement: Tenure, Purchase history, Login frequencies, Last Activity timelines.
- Sentiment Metrics: Customer Support Requests, UX Satisfaction score.
- Output Objective: Churn Status (Binary Target).

## 2. Preprocessing & Feature Engineering
- Feature Distillation: Raw calendar indices representing the final operational contacts were transformed mathematically into numerical relative distances: `Days_Since_Last_Activity = Current_Date - Last_Activity_Date`.
- Categorical Vectorization: Labels for textual categorical constraints like Country/Gender classifications were parsed explicitly via Label Encoding maps.
- Feature Normalization: Continuous scale data matrices were standard-scaled to prevent distance-bias metrics skewing the weights.
- Stratified Split: Subsets segregated systematically using an 80/20 train-to-test partitioning routine.

## 3. Results & Evaluation Matrix
Two major tree-based frameworks were verified against the blind testing segment:
- XGBoost: Achieved the highest performance across Precision, Recall, and ROC-AUC. 
- Random Forest: Demonstrated robust stability but slight over-indexing on historical edge rules compared to gradients.

## 4. Operational Business Insights
- High churn risks correlate explicitly with customers exceeding 5 service intervention logs combined with multi-week platform inactivity.
- Actionable Strategy: Setup a micro-automation rule targeting accounts reaching 4 or more Customer Support Requests to issue targeted discount structures before churn conditions peak.