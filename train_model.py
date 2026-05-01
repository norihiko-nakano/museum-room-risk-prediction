import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv("museum_room_risk_data.csv")

y = df["actual_incident"]

# Model 1: visitor_count only
X1 = df[["visitor_count"]]

# Model 2: visitor_count + room_id
X2 = df[["visitor_count", "room_id"]]


def evaluate_model(name, X, y, use_room_id=False):
    if use_room_id:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["visitor_count"]),
                ("cat", OneHotEncoder(drop=None), ["room_id"]),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["visitor_count"]),
            ]
        )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"=== {name} ===")
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


results = []

results.append(evaluate_model(
    "Model 1: Visitor Count Only",
    X1,
    y,
    use_room_id=False
))

results.append(evaluate_model(
    "Model 2: Visitor Count + Room ID",
    X2,
    y,
    use_room_id=True
))

result_df = pd.DataFrame(results)
result_df.to_csv("results.csv", index=False, encoding="utf-8-sig")

print("Created results.csv")
print(result_df)