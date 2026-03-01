# Credit Default Prediction using XGBoost


## Problem Statement


In this project we build a machine learning model to estimate Probability of Default (PD) on an imbalance credit dataset. The objective is to develop a robust classification model capable of ranking borrower risk and supporting credit approval.




## Dataset Overview


* Binary target variable
* Highly imbalance date with ~ 9% positive class, 91% negative class
* Contains information about borrower credit scores
* Includes external credit risk scores and socioeconomic indicators




## Project Workflow


The project is structured across three notebooks


1. EDA.ipynb : distribution analysis, missing value assessment, class imbalance evaluation, and initial signal inspection through correlation analysis.
2. Data_Preprocessing_and_Baseline_Models.ipynb : Data cleaning, missing value handling, categorical encoding, and benchmarking of Logistic Regression, Random Forest, and XGBoost using stratified cross-validation.
3. Final_Model_XGBoost.ipynb : Hyperparameter tuning with RandomizedSearchCV, leakage-safe threshold optimization using cross-validated predictions, final model evaluation, feature importance analysis, and SHAP-based interpretability.


## Modeling Approach
* Selection of best performing baseline model on the basis of performance metrics, especially ROC AUC score and PR AUC score.
* Stratified 80/20 train test split
* Hyperparameter tuning via RandomizedSearchCV (5-fold cross-validation)
* Threshold selection using cross-validated training probabilities and performance metrics like F1 score to avoid data leakage
* Evaluation using ROC-AUC and PR-AUC (appropriate for imbalanced classification)


## Final Results

| Metric  | Cross-Validation | Test Set |
|---------|------------------|----------|
| ROC-AUC | 0.756            | 0.760    |
| PR-AUC  | 0.239            | 0.251    |	
	

The close alignment between cross-validation and test performance indicates strong generalization and minimal overfitting.


## Key Insights


* Feature importance analysis indicates that risk indicators provided by external sources (EXT_SOURCE 2,3) act as the most important features. The model correctly captures historical credit behaviour of borrowers as top contenders for default prediction.
* Level of education and employment history are accurately flagged as important contributing features. Socioeconomic variables like whether borrowers own a car, where and whom they live with, number of elevators in their house etc show up as wealth proxies.
* SHAP analysis shows that lower values of EXT SOURCE (credit score) pushes risk up or pushes probability of default closer to one. However lower credit value amount indicates lower probability of default.
* This dataset shows that lower levels of education lowers probability of default. It could be possible that borrowers with higher educational levels avail bigger loans so are at a higher risk of default, but it needs to be studied further.
* It also shows up here that lower DAYS_EMPLOYED reduces probaility of default. DAYS_EMPLOYED has negative values, starting from the day the loan has been taken to the start of employment. So once that is accounted for, we see that better employment history reduces risk of default which is in tune with the generalised situation.



## Interpretation & Business Relevance
The model demonstrates strong ranking ability and meaningful minority-class detection, making it suitable for credit risk prioritization and decision support.
While threshold selection in production would typically be business cost-sensitive, this implementation provides a statistically sound modeling and evaluation framework.