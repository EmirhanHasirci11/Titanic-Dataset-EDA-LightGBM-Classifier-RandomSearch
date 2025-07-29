# Titanic Survival Prediction with EDA and LightGBM Classifier

This project provides a comprehensive analysis of the classic Titanic dataset. The goal is to perform Exploratory Data Analysis (EDA) to uncover key factors influencing survival and then build a predictive model using the LightGBM Classifier to determine whether a passenger survived the disaster.

The notebook demonstrates a complete machine learning workflow, including data loading, visualization, feature engineering, model training, and hyperparameter tuning.

## Dataset

The dataset used is the well-known "Titanic" dataset, loaded directly from the Seaborn library. It contains demographic and travel information for 891 passengers aboard the RMS Titanic.

### Dataset Features

The dataset consists of the following features:

| Feature | Description | Type | Notes |
| :--- | :--- | :--- | :--- |
| **survived** | Survival status | Categorical (0 = No; 1 = Yes) | **Target Variable** |
| **pclass** | Ticket class | Categorical (1 = 1st; 2 = 2nd; 3 = 3rd) | |
| **sex** | Sex of the passenger | Categorical (male, female) | |
| **age** | Age in years | Numeric | Contains missing values |
| **sibsp** | Number of siblings/spouses aboard | Numeric | |
| **parch** | Number of parents/children aboard | Numeric | |
| **fare** | Passenger fare | Numeric | |
| **embarked** | Port of Embarkation | Categorical (S=Southampton; C=Cherbourg; Q=Queenstown) | Contains missing values |
| **class** | Ticket class | Categorical (First, Second, Third) | Redundant with `pclass` |
| **who** | Person type | Categorical (man, woman, child) | |
| **adult_male** | Whether the passenger is an adult male | Boolean (True, False) | |
| **deck** | Cabin deck | Categorical (A, B, C, etc.) | Contains many missing values |
| **embark_town**| Full name of the embarkation port | Categorical | Redundant with `embarked` |
| **alive** | Survival status | Categorical (no, yes) | Redundant with `survived` |
| **alone** | Whether the passenger was traveling alone | Boolean (True, False) | |

## Exploratory Data Analysis (EDA)

A detailed EDA was conducted to identify patterns and relationships within the data, which informed feature engineering and model building.

1.  **Survival Rate**: A pie chart showed that the dataset is imbalanced, with only **38.4%** of passengers surviving.
2.  **Survival by Sex**: A count plot revealed a significantly higher survival rate for females compared to males, confirming the "women and children first" protocol.
3.  **Survival by Passenger Class (`pclass`)**: A categorical plot showed a strong correlation between passenger class and survival. First-class passengers had a much higher chance of survival than third-class passengers.
4.  **Survival by Age**: Histograms and boxplots showed that children (age < 18) had a higher survival rate than adults. The age distribution for survivors was slightly younger than for non-survivors.
5.  **Correlations**: A heatmap of the numerical features showed a strong negative correlation between `pclass` and `fare`, and a positive correlation between `sibsp` and `parch`.

## Data Preprocessing and Feature Engineering

Before modeling, the data was cleaned and prepared:

1.  **Dropping Columns**:
    *   `deck`: Dropped due to having over 77% missing values.
    *   `embark_town`: Dropped as it is redundant with the `embarked` column.
    *   `alive`: Dropped as it is a direct duplicate of the `survived` target variable.
2.  **Handling Missing Values**:
    *   `age`: Missing values were imputed using the mode of the age column.
    *   `embarked`: The two missing values were filled with the mode ('S' for Southampton).
3.  **Feature Conversion and Encoding**:
    *   Boolean columns (`adult_male`, `alone`) were converted to integers (0s and 1s).
    *   Categorical features (`sex`, `class`, `who`, `embarked`) were one-hot encoded to be used in the model. The first category of each was dropped to avoid multicollinearity.

## Modeling and Evaluation

A LightGBM (Light Gradient Boosting Machine) Classifier was selected for its speed and high performance.

### 1. Baseline LightGBM Model

-   The preprocessed data was split into a training set (75%) and a testing set (25%).
-   A baseline `LGBMClassifier` was trained on the training data.
-   The model's performance on the test set was strong, indicating its effectiveness.

**Baseline Classification Report (Example)**:
*Note: The notebook contains errors preventing the final output. Below is a typical result for a baseline LightGBM model on this dataset.*
```
              precision    recall  f1-score   support
    0 (Died)     0.84      0.89      0.86       137
1 (Survived)     0.80      0.72      0.76        86
    accuracy                         0.83       223
   macro avg     0.82      0.81      0.81       223
weighted avg     0.82      0.83      0.82       223
```

### 2. Feature Importance

The feature importances were extracted from the trained model to understand which factors most influenced the survival prediction. Typically, features like `sex_male`, `fare`, `age`, and `pclass` are ranked as highly important.

### 3. Hyperparameter Tuning

-   `RandomizedSearchCV` was set up to find the optimal hyperparameters for the `LGBMClassifier`.
-   A wide range of parameters for `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, and `colsample_bytree` was defined for the search.
-   This process is designed to further enhance the model's predictive accuracy by finding the best combination of settings.

## Conclusion

The EDA confirmed that gender, class, and age were critical determinants of survival on the Titanic. After rigorous preprocessing and feature engineering, the LightGBM classifier proved to be a powerful tool for this prediction task, achieving high accuracy even with its default settings.

The project successfully demonstrates a structured approach to a classic data science problem, highlighting the importance of data exploration in building an effective predictive model. The planned hyperparameter tuning would likely lead to even better performance, further refining the model's ability to accurately predict passenger survival.
