# Titanic Survival Prediction: an End-to-End ML Pipeline

A complete supervised learning workflow on the Titanic passenger data, framed for a travel insurance and maritime risk company exploring whether booking information alone can estimate passenger risk.

**[View the full notebook with all outputs](https://ayushramtekkar.github.io/titanic-survival-ml-pipeline/)**

## Approach

- **Data:** Titanic passenger dataset (891 rows) from the [Data Science Dojo datasets repository](https://github.com/datasciencedojo/datasets)
- **Exploratory analysis** of class, gender, age, family size and fare
- **Feature engineering:** title extracted from names, family size, imputation for missing values
- **Preprocessing comparison:** an empirical test of scaling versus no scaling with 5-fold cross-validation
- **Models:** Logistic Regression and Random Forest in scikit-learn `Pipeline` / `ColumnTransformer`
- **Tuning:** `GridSearchCV` with 5-fold cross-validation, plus learning curves

## Results (held-out test set, 179 passengers)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression** (C = 10) | **0.84** | **0.83** |
| Random Forest (max_depth = 4, 100 trees) | 0.81 | 0.79 |

- Gender, passenger class and title are the most influential features.
- Logistic Regression wins and is also easier to explain to non-technical stakeholders.
- The model is suitable as a reference tool to support human judgement, not for automated decisions, given the small and historically specific dataset.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib

## Notes

Completed as part of the MSc Data Science, AI and Digital Business, Gisma University of Applied Sciences.
