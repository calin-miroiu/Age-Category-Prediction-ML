# Age Category Predictor: Lifestyle & Financial Analysis

## Overview
This project implements an end-to-end Machine Learning pipeline to predict an individual's age category based on specific lifestyle and financial indicators. 

Developed as part of the Computer Engineering curriculum at University Politehnica of Bucharest, the project demonstrates proficiency in data simulation, preprocessing, Exploratory Data Analysis (EDA), and predictive modeling using Python.

## Key Features & Technical Pipeline

### 1. Context-Aware Synthetic Data Generation
Since no existing datasets perfectly matched the specific behavioral rules required, a custom dataset of 800 individuals was synthetically generated using `numpy` and `pandas`[cite: 10]. 
* Data was not generated purely at random; instead, Gaussian distributions were applied with different parameters for each age group to mimic real-world demographic behaviors (e.g., middle-aged individuals exhibit higher incomes but elevated stress levels, while students have lower incomes and more free time).
* **Features included:** Income, Savings, Daily Screen Time, Stress Level, Free Time, Occupation, and Marital Status.

### 2. Data Cleaning & Preprocessing
To simulate messy, real-world data environments, intentional anomalies were introduced and systematically handled:
* **Missing Values:** Induced null values in the "Daily Screen Time" column, which were successfully imputed using the median to avoid skewing the distribution.
* **Outlier Handling:** Deliberately injected extreme values (e.g., a stress level of 12 on a 1-10 scale) and capped them dynamically using `np.where`.

### 3. Exploratory Data Analysis (EDA)
Comprehensive visual analysis was conducted using `seaborn` and `matplotlib` to validate the logical rules of the dataset:
* **Histograms & Bar Charts:** Verified class balance across the 4 age categories (18-25, 26-39, 40-59, 60+).
* **Boxplots:** Highlighted outliers and illustrated the clear correlation between active professional age groups and higher median incomes.
* **Heatmaps:** Confirmed the absence of redundant, perfectly correlated numerical variables, ensuring all features added value to the model.

### 4. Predictive Modeling
* **Algorithm:** `RandomForestClassifier` from `scikit-learn`.
* **Performance:** The model achieved an accuracy of ~99%. 
* **Evaluation Insight:** While a near-perfect score often indicates overfitting in real-world scenarios, it is mathematically justified here. Because the dataset was generated using strict probability distributions and logical constraints, the classes are perfectly separable. The Random Forest algorithm successfully reverse-engineered the exact mathematical rules used during the data generation phase[cite: 15].

## Technologies Used
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`

## Requirements
- Python 3.8+
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn
