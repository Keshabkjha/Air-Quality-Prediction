# Air-Quality-Prediction

This repository contains an Air Quality predictive model that leverages advanced machine learning techniques to predict the Air Quality Index (AQI). The project incorporates robust data imputation, model training, and evaluation steps, ensuring the highest possible accuracy and reliability.

## Table of Contents
- [Project Overview](#project-overview)
- [Techniques Used](#techniques-used)
- [Data Preprocessing](#data-preprocessing)
  - [Handling Missing Data](#handling-missing-data)
  - [Outlier Removal](#outlier-removal)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Visualization](#visualization)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [License](https://github.com/Keshabkjha/Air-Quality-Prediction/blob/main/LICENSE)
- [Kaggle Notebook & Dataset](https://www.kaggle.com/code/keshabkkumar/air-quality-prediction)

## Project Overview

This project aims to predict the Air Quality Index (AQI) using a series of machine learning techniques, focusing on accurate handling of missing data and optimal model selection.

## Techniques Used

1. **Multiple Imputation by Chained Equations (MICE)**:
   - Imputation techniques: Ridge Regression, Bayesian Regression, Elastic Net
   - Used to efficiently handle missing values in the dataset.

2. **Outlier Detection and Removal**:
   - Outliers were eliminated prior to imputation to avoid skewing the model results.

3. **Machine Learning Models**:
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor

## Data Preprocessing

### Handling Missing Data

We tackled missing data using the **Multiple Imputation by Chained Equations (MICE)** technique, implemented via Scikit-learn's `IterativeImputer`. This method allowed us to impute missing values based on relationships between different features.

Estimators used for imputation:
- **Ridge Regression**
- **Bayesian Regression**
- **Elastic Net**

### Outlier Removal

Before imputation, outliers were removed to ensure they did not distort the dataset and the resulting imputation models. This preprocessing step significantly improved the model's performance and reliability.

## Models Implemented

- **Decision Tree Regressor**: A simple tree-based model for regression tasks.
- **Random Forest Regressor**: An ensemble method based on multiple decision trees, providing robust predictions.
- **Gradient Boosting Regressor**: A boosting method that builds models sequentially to reduce error iteratively.

## Results

After testing the dataset with these models, the **Random Forest Regressor** provided the most accurate predictions for the Air Quality Index (AQI), outperforming other models.

## Visualization

Each model's performance was visualized, and comparisons were made between models based on:
- Ridge Regression
- Bayesian Regression
- Elastic Net

These visualizations helped to identify the most effective model, with Random Forest emerging as the top performer.

## How to Run

Clone the repository:
   ```bash
   git clone https://github.com/Keshabkjha/Air-Quality-Prediction.git
   cd Air-Quality-Prediction
  ```
## Dependencies
  • Python 3.x
  • Scikit-learn
  • Pandas
  • NumPy
  • Matplotlib
  • Seaborn
## License 
  This project is licensed under the MIT License. See the LICENSE (https://www.kaggle.com/code/keshabkkumar/air-quality-prediction) file for details.

## Kaggle Notebook & Dataset
  The notebook and dataset for this project are available on Kaggle:
  Kaggle Notebook and Dataset (https://www.kaggle.com/code/keshabkkumar/air-quality-prediction)
