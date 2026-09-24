# Student Performance Analysis and Machine Learning

## Overview

This project analyzes student performance data using exploratory data analysis (EDA), data cleaning, feature engineering, machine learning, cross-validation, and hyperparameter tuning.

The main objective is to build a machine learning model that predicts student final-grade categories using academic, attendance, demographic, and study-related features.

---

## Project Objectives

- Understand the structure and quality of the student performance dataset
- Clean and prepare the data for machine learning
- Perform exploratory data analysis
- Engineer useful machine learning features
- Investigate potential target leakage
- Compare multiple classification models
- Evaluate models using multiple performance metrics
- Perform k-fold cross-validation
- Tune the final model's hyperparameters
- Analyze feature importance
- Save the final trained model and results

---

## Project Workflow

The project follows this workflow:

1. Data inspection
2. Data cleaning
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Target leakage investigation
6. Machine learning experiments
7. Model comparison
8. K-fold cross-validation
9. Hyperparameter tuning
10. Final model training
11. Feature importance analysis
12. Final results and reporting

---

## Dataset

The project uses student performance data containing academic, demographic, attendance, and study-related variables.

Important features include:

- Study hours
- Mathematics score
- Science score
- English score
- Attendance percentage
- Age
- Parent education
- Travel time
- Extra activities
- Internet access
- Study method
- Gender
- School type

The project contains cleaned and machine-learning-ready versions of the dataset.

---

## Machine Learning

Several classification approaches were investigated during the project, including:

- Logistic Regression
- Random Forest
- Gradient Boosting

Model performance was evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score

---

## Model Comparison

The final model comparison produced the following results:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6717 | 0.6452 | 0.6068 | 0.6201 |
| Random Forest | 0.7623 | 0.7683 | 0.7391 | 0.7518 |
| Final Tuned Random Forest | 0.9117 | 0.9225 | 0.9043 | 0.9127 |

---

## Final Model

The final selected model is a **tuned Random Forest classifier**.

The trained model is saved in:

```text
ML_Results/final_tuned_random_forest.joblib