# Car-Evaluation-Using-Machine-Learning-
#block 1
# Install kaggle and a compatible version of scikit-learn
!pip install -q kaggle
!pip install -q scikit-learn==1.6.1
#Block 2
import pandas as pd

# Load the uploaded CSV file
df = pd.read_csv('/content/car_evaluation.csv', header=None)
df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

# Display first few rows
df.head()
#Block 3
# Visualize relationship between safety and class
plt.figure(figsize=(6, 4))
sns.countplot(x="safety", hue="class", data=df, palette="Set2")
plt.title("Safety Level vs Class")
plt.xlabel("Safety")
plt.ylabel("Count")
plt.legend(title="Class")
plt.show()

# Visualize buying vs class
plt.figure(figsize=(6, 4))
sns.countplot(x="buying", hue="class", data=df, palette="Set3")
plt.title("Buying Price vs Class")
plt.xlabel("Buying Price")
plt.ylabel("Count")
plt.legend(title="Class")
plt.show()
#Block 4
from sklearn.preprocessing import LabelEncoder

# Encode all categorical columns
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target
X = df.drop("class", axis=1)
y = df["class"]
#Block 5
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Block 6
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Results:\n")
print(classification_report(y_test, y_pred_lr))
#Block 7
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("k-NN Results:\n")
print(classification_report(y_test, y_pred_knn))
#Block 8
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Results:\n")
print(classification_report(y_test, y_pred_dt))
#Block 9
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='weighted')
    ]

models = ["Logistic Regression", "k-NN", "Decision Tree"]
metrics = np.array([
    get_metrics(y_test, y_pred_lr),
    get_metrics(y_test, y_pred_knn),
    get_metrics(y_test, y_pred_dt)
])

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

for i, name in enumerate(metric_names):
    plt.figure()
    plt.bar(models, metrics[:, i], color=["skyblue", "orange", "lightgreen"])
    plt.title(name + " Comparison")
    plt.ylabel(name)
    plt.ylim(0, 1)
    plt.show()
#Block 10
# Use this markdown content in a text cell in your notebook for documentation:
"""
### Model Comparison Summary

**Logistic Regression**:
- Pros: Fast, interpretable.
- Cons: Struggles with non-linear data.
- F1 Score: Moderate.

**k-Nearest Neighbors**:
- Pros: Easy to understand, no training time.
- Cons: Slow with large datasets, sensitive to noise.
- F1 Score: Good.

**Decision Tree**:
- Pros: Interpretable, handles complex data well.
- Cons: Can overfit without pruning.
- F1 Score: Best among all three.

### Conclusion:
Decision Tree gave the best balance of accuracy and interpretability for this automobile classification task.
"""
#Block 11
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Collect predictions for all models
models = {
    "Logistic Regression": y_pred_lr,
    "k-Nearest Neighbors": y_pred_knn,
    "Decision Tree": y_pred_dt
}

# Evaluate each model
comparison_metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}

for name, preds in models.items():
    comparison_metrics["Model"].append(name)
    comparison_metrics["Accuracy"].append(accuracy_score(y_test, preds))
    comparison_metrics["Precision"].append(precision_score(y_test, preds, average='weighted', zero_division=0))
    comparison_metrics["Recall"].append(recall_score(y_test, preds, average='weighted'))
    comparison_metrics["F1-Score"].append(f1_score(y_test, preds, average='weighted'))

# Convert to DataFrame
metrics_df = pd.DataFrame(comparison_metrics)

# Descriptive summary
print("=== Descriptive Statistics for Model Metrics ===")
print(metrics_df.describe())

# Best-performing model per metric
best_models = metrics_df.set_index("Model").idxmax()
print("\n=== Best Model for Each Metric ===")
print(best_models)

# Visual comparison using Seaborn
plt.figure(figsize=(10, 6))
melted_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette="muted")
plt.title("Model Comparison on Evaluation Metrics")
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

