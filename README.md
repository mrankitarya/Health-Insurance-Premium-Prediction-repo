# Insurance Premium Prediction

## Overview
This project predicts insurance premiums based on user input such as age, BMI, number of children, smoking habits, gender, and region. It utilizes machine learning models to provide accurate predictions and includes a Streamlit-based user interface for easy interaction.

## Features
- Data preprocessing pipelines for both numerical and categorical features.
- Implementation of multiple regression models:
  - Simple Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
  - Gradient Boosting Regressor
- Performance evaluation using metrics such as R², Mean Absolute Error, and Mean Squared Error.
- Interactive Streamlit app for user input and prediction.

## Dataset
The dataset used in this project is `insurance.csv`, which includes the following features:
- `age`: Age of the primary beneficiary.
- `sex`: Gender of the beneficiary.
- `bmi`: Body Mass Index (BMI).
- `children`: Number of children/dependents covered by health insurance.
- `smoker`: Smoking status (yes/no).
- `region`: Residential region in the US (southwest, southeast, northwest, northeast).
- `expenses`: Individual medical costs billed by health insurance.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Processing: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn
  - Deployment: Streamlit


## Streamlit Interface
Here is a snapshot of the Streamlit interface:

![Streamlit Interface](C:\Users\GABRU\Pictures\Screenshots\Screenshot 2025-01-22 123714.png)


