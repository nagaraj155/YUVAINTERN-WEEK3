# Week 3 Task: Unsupervised Learning & Model Evaluation

## 1. Introduction

This project demonstrates unsupervised learning techniques
and machine learning model evaluation using the Iris dataset.

The main objective is to understand clustering algorithms,
dimensionality reduction, model evaluation techniques,
cross-validation, and hyperparameter tuning.

---

## 2. Objectives

- Understand K-Means Clustering
- Understand Hierarchical Clustering
- Apply dimensionality reduction using PCA
- Visualize clusters
- Evaluate a classification model
- Calculate Accuracy, Precision, Recall and F1-score
- Generate a Confusion Matrix
- Calculate ROC-AUC
- Perform Cross-Validation
- Perform Hyperparameter Tuning

---

## 3. Dataset

The Iris dataset from Scikit-learn was used.

The dataset contains:

- 150 samples
- 4 features
- 3 target classes

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Classes

- Setosa
- Versicolor
- Virginica

---

## 4. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 5. Data Preprocessing

The Iris dataset was loaded using Scikit-learn.

StandardScaler was used to standardize the features before
performing clustering.

Standardization ensures that all features are on a comparable
scale.

---

## 6. K-Means Clustering

K-Means clustering was applied to divide the dataset into
three clusters.

The Elbow Method was used to determine a suitable number
of clusters.

For the Iris dataset, K = 3 was selected.

### Output

The Elbow Method graph and K-Means cluster visualization
are stored in the outputs folder.

---

## 7. Hierarchical Clustering

Agglomerative Hierarchical Clustering was applied with
three clusters.

A dendrogram was created using the Ward linkage method
to visualize the hierarchical relationships between
data points.

---

## 8. PCA Dimensionality Reduction

Principal Component Analysis (PCA) was used to reduce
the four-dimensional dataset into two principal components.

This made it easier to visualize the data and clusters
in a two-dimensional graph.

---

## 9. Classification Model

Logistic Regression was used as a supervised learning model
to demonstrate model evaluation techniques.

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

---

## 10. Model Evaluation

The following evaluation metrics were calculated:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many predicted positive samples are actually
positive.

### Recall

Measures how many actual positive samples are correctly
identified.

### F1 Score

F1-score is the harmonic mean of precision and recall.

### ROC-AUC

ROC-AUC measures the ability of the classifier to distinguish
between different classes.

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect
predictions for each class.

---

## 11. Cross-Validation

5-fold cross-validation was performed to evaluate the
consistency of the Logistic Regression model.

The dataset was divided into five parts and the model was
trained and evaluated multiple times.

---

## 12. Hyperparameter Tuning

GridSearchCV was used to find the best parameters for the
Logistic Regression model.

The following parameters were tested:

- C
- Solver

The best combination was selected based on cross-validation
accuracy.

---

## 13. Results

The final model evaluation results are obtained from the
Python program.

| Metric | Result |
|---|---|
| Accuracy | Add your result |
| Precision | Add your result |
| Recall | Add your result |
| F1 Score | Add your result |
| ROC-AUC | Add your result |
| Mean CV Accuracy | Add your result |
| Best CV Score | Add your result |

### Best Parameters

Add the parameters displayed by the program.

---

## 14. Output Files

The following visualizations were generated:

- Elbow Curve
- K-Means Clusters
- Hierarchical Clustering
- Dendrogram
- PCA Visualization
- Confusion Matrix

---

## 15. Conclusion

In this project, K-Means and Hierarchical Clustering were
successfully implemented on the Iris dataset.

PCA was used for dimensionality reduction and visualization.

Logistic Regression was evaluated using Accuracy, Precision,
Recall, F1-score, Confusion Matrix and ROC-AUC.

5-fold Cross-Validation was performed to measure model
reliability, and GridSearchCV was used for hyperparameter
tuning.

The project provided practical understanding of both
unsupervised learning and machine learning model evaluation.