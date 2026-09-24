# Student Performance Analysis and Machine Learning

## Overview

This project analyzes student performance data using exploratory data
analysis, feature engineering, machine learning, cross-validation, and
hyperparameter tuning.

The goal is to develop a machine learning model capable of predicting
student final-grade categories from academic and demographic features.

---

## Project Workflow

The project follows this workflow:

1. Data inspection
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Machine learning experiments
6. Model comparison
7. Cross-validation
8. Hyperparameter tuning
9. Final model training
10. Final findings report

---

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Experiments

### Experiment A

Experiment A includes `overall_score` as a feature.

This experiment produces very high performance because the overall
score is closely related to the final grade.

### Experiment B

Experiment B uses subject-level scores and other available features
without relying on `overall_score`.

This provides a more realistic assessment of prediction performance.

---

## Final Model

The final model is a tuned Gradient Boosting classifier.

Hyperparameters:

- `learning_rate = 0.05`
- `max_depth = 3`
- `n_estimators = 200`
- `subsample = 0.8`
- `random_state = 42`

---

## Final Performance

For Experiment B, the tuned Gradient Boosting model achieved
approximately:

- Accuracy: 76.37%
- Macro F1: 76.44%

Five-fold cross-validation produced:

- Mean Macro F1: 76.44%
- Standard deviation: 0.42 percentage points

---

## Project Outputs

### EDA Charts

The `EDA_Charts` directory contains visualizations including:

- Parent education vs overall score
- Study hours vs overall score
- Attendance vs overall score
- Correlation matrix
- Grade by study method
- Overall score by study method

### Machine Learning Results

The `ML_Results` directory contains:

- Model comparison results
- Cross-validation results
- Hyperparameter tuning results
- Feature importance results
- Final model results
- Complete findings report

---

## Final Model File

The trained model is saved as:

`final_gradient_boosting_model.joblib`

---

## Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Machine Learning
- Exploratory Data Analysis

---

## Conclusion

The project compares multiple classification approaches and evaluates
their performance using accuracy, precision, recall, F1-score,
cross-validation, and hyperparameter tuning.

The final analysis provides a reproducible machine learning workflow
for student performance classification.