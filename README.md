# Pokémon Legendary Predictor

**Pokémon Legendary Predictor** is a machine learning project that uses a K-Nearest Neighbors (KNN) classifier to predict whether a Pokémon is "Legendary" based on its base stats and primary type. The project features a robust, automated Scikit-Learn preprocessing pipeline, including custom text sanitization, missing value imputation, and hyperparameter tuning.

## 🚀 Features
* **Custom Scikit-Learn Transformer:** Implements a custom `TextCleaner` class to sanitize, lowercase, and strip text data while preserving missing values.
* **Automated Data Pipeline:** Utilizes a `ColumnTransformer` to seamlessly handle mixed data types—applying One-Hot Encoding to categorical columns and Standard Scaling to numerical stats.
* **Stratified Cross-Validation:** Uses 10-fold Stratified Cross-Validation to ensure robust performance metrics despite the highly imbalanced nature of "Legendary" Pokémon data.
* **Hyperparameter Optimization:** Leverages `GridSearchCV` to automatically test and identify the optimal number of neighbors ($k$) for the KNN classifier.

## 🛠️ Tech Stack
* **Language:** Python (Quarto Notebook)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`sklearn`)

## 📋 Pipeline Workflow
1. **Data Cleaning:** Drop redundant columns (`#`, `Name`, `Type 2`, `Total`, `Generation`) and remove duplicate records.
2. **Preprocessing Setup:** * **Categorical (`Type 1`):** Cleansed $\rightarrow$ Imputed (most frequent) $\rightarrow$ One-Hot Encoded.
   * **Numerical (Stats):** Imputed (median) $\rightarrow$ Standardized using `StandardScaler`.
3. **Hyperparameter Tuning:** Search across a grid of $k = 1$ to $49$ using `GridSearchCV`.
4. **Final Evaluation:** The best estimator is extracted and evaluated against a isolated 30% test split.

## 📦 Requirements & Getting Started

To run this Quarto notebook (`.qmd`), ensure you have Python and the required libraries installed:

```bash
pip install pandas numpy scikit-learn
