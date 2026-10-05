# Week 3 Task: Unsupervised Learning & Model Evaluation
import numpy as np
import pandas as pd
import os

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing required dependency. Install the project packages with: "
        "pip install numpy pandas matplotlib seaborn scikit-learn"
    ) from exc

from scipy.cluster.hierarchy import dendrogram, linkage


# Create output folder
os.makedirs("outputs", exist_ok=True)


# =========================================================
# 1. LOAD DATASET
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)

print("\n===== DATASET =====")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nTarget Classes:")
print(target_names)


# =========================================================
# 2. DATA PREPROCESSING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n===== DATA PREPROCESSING =====")
print("Features standardized successfully.")


# =========================================================
# 3. K-MEANS CLUSTERING
# =========================================================

inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


# Elbow Curve
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")

plt.savefig("outputs/elbow_curve.png")

plt.show()


# K-Means with K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

print("\n===== K-MEANS =====")
print("Cluster Centers:")
print(kmeans.cluster_centers_)

print("\nCluster Counts:")
print(pd.Series(kmeans_labels).value_counts())


# =========================================================
# 4. PCA FOR VISUALIZATION
# =========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n===== PCA =====")
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Explained Variance:",
    sum(pca.explained_variance_ratio_)
)


# K-Means visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering using PCA")

plt.savefig("outputs/kmeans_clusters.png")

plt.show()


# =========================================================
# 5. HIERARCHICAL CLUSTERING
# =========================================================

linked = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(10, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")

plt.savefig("outputs/dendrogram.png")

plt.show()


# Agglomerative Clustering
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

print("\n===== HIERARCHICAL CLUSTERING =====")

print("Cluster Counts:")
print(pd.Series(hierarchical_labels).value_counts())


# Hierarchical visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_labels,
    cmap="plasma",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Hierarchical Clustering using PCA")

plt.savefig("outputs/hierarchical_clusters.png")

plt.show()


# =========================================================
# 6. PCA CLUSTER VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Actual Iris Classes using PCA")

plt.colorbar(scatter)

plt.savefig("outputs/pca_clusters.png")

plt.show()


# =========================================================
# 7. SUPERVISED MODEL
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=200
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)


# =========================================================
# 8. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

roc_auc = roc_auc_score(
    y_test,
    y_probability,
    multi_class="ovr"
)


print("\n===== MODEL EVALUATION =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )
)


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()


# =========================================================
# 10. CROSS VALIDATION
# =========================================================

cv_scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n===== CROSS VALIDATION =====")

print("CV Scores:", cv_scores)

print(
    "Mean CV Accuracy:",
    cv_scores.mean()
)


# =========================================================
# 11. HYPERPARAMETER TUNING
# =========================================================

parameters = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=500),
    parameters,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_scaled, y)


print("\n===== HYPERPARAMETER TUNING =====")

print(
    "Best Parameters:",
    grid_search.best_params_
)

print(
    "Best CV Score:",
    grid_search.best_score_
)


# =========================================================
# 12. FINAL RESULT
# =========================================================

print("\n===================================")
print("       WEEK 3 TASK COMPLETED")
print("===================================")

print("K-Means Clustering       : Completed")
print("Hierarchical Clustering  : Completed")
print("PCA Dimensionality       : Completed")
print("Confusion Matrix         : Completed")
print("Precision                : Completed")
print("Recall                   : Completed")
print("F1 Score                 : Completed")
print("ROC-AUC                  : Completed")
print("Cross Validation         : Completed")
print("Hyperparameter Tuning    : Completed")
