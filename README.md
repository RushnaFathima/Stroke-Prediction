# 🧠 Stroke Prediction Project

## Overview
This project predicts whether a person is likely to have a stroke based on health and lifestyle data.
It uses machine learning to analyze patterns in the dataset and make predictions that can help with early diagnosis.

## Objective
Stroke is one of the leading causes of death and disability worldwide. Early prediction can help save lives.
The main objective of this project is to build a predictive model that can classify whether a patient is at risk of stroke based on various input parameters such as:
- Age
- Hypertension
- Heart disease
- BMI
- Average glucose level
- Gender
- Smoking status
- Work type
- residence type
## Tools and Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Modelling**: Decision Tree, Random Forest, XGBoost
- **Visualization**: Matplotlib & Seaborn
- **Deployment**: Streamlit
## Dataset
The dataset used is from the Kaggle Stroke Prediction Dataset.
It contains 5110 rows and 12 columns, providing patient information and a target variable indicating whether a patient has suffered a stroke.

| Column            | Description                                |
| ----------------- | ------------------------------------------ |
| gender            | Gender of the patient                      |
| age               | Age of the patient                         |
| hypertension      | 0 if no hypertension, 1 if yes             |
| heart_disease     | 0 if no heart disease, 1 if yes            |
| ever_married      | Marital status                             |
| work_type         | Type of occupation                         |
| Residence_type    | Urban or Rural                             |
| avg_glucose_level | Average glucose level                      |
| bmi               | Body mass index                            |
| smoking_status    | Current smoking habits                     |
| stroke            | 1 if the patient had a stroke, 0 otherwise |


An analysis of the target variable distribution revealed a significant class imbalance:

| **Class** | **Count** | **Description**      |
| :-------: | --------: | :------------------- |
| 0         | 4861      | Non-stroke cases     |
| 1         | 249       | Stroke cases         |


## Results
| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------- | -------: | --------: | -----: | -------: | ------: |
| Decision Tree |   90.54% |    11.11% | 13.33% |   12.12% |  53.92% |
| Random Forest |   94.59% |    21.43% |  4.00% |    6.74% |  51.62% |
| XGBoost       |   65.62% |    10.76% | 82.67% |   19.05% |  73.71% |

**Key Insights:**
- Decision Tree and Random Forest models achieved high accuracy (90.54% and 94.59%, respectively), their precision and recall values are very low, indicating that these models mostly predicted the majorityclass (non-stroke) correctly but performed poorly on the minority class(stroke).
- XGBoost model achieved a lower overall accuracy (65.62%) but a significantly higher recall (82.67%) and ROC-AUC score (73.71%).


In high-risk medical tasks like stroke prediction,recall and ROC-AUC score are more important than raw accuracy.Therefore, XGBoost is the most suitable model for this task, as it identifies the majority of stroke cases despite lower overall accuracy.
