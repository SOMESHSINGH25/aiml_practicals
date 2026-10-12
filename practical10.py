
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample multimodal dataset
data = {
    "text": [
        "Government announces new education policy",
        "Scientists publish climate research findings",
        "Health department issues vaccination advisory",
        "University announces examination schedule",
        "Official report releases economic statistics",
        "City administration announces road repairs",
        "Breaking secret cure discovered overnight",
        "Celebrity confirms unbelievable miracle claim",
        "Shocking story claims impossible medical results",
        "Viral post alleges secret worldwide conspiracy",
        "Fake message promises instant wealth",
        "Unverified report claims impossible events",
        "Research institute publishes study results",
        "Election commission releases official notice",
        "Hospital publishes public health guidelines",
        "Weather department issues storm warning",
        "Forwarded message claims magical treatment",
        "Anonymous post announces extraordinary discovery",
        "Official agency publishes annual report",
        "Verified notice announces public examination"
    ],
    "brightness": [
        0.5, 0.6, 0.4, 0.5, 0.6,
        0.4, 0.9, 0.8, 0.9, 0.8,
        0.9, 0.8, 0.5, 0.6, 0.4,
        0.5, 0.9, 0.8, 0.5, 0.6
    ],
    "edge_density": [
        0.3, 0.4, 0.3, 0.4, 0.3,
        0.4, 0.9, 0.8, 0.9, 0.8,
        0.9, 0.8, 0.3, 0.4, 0.3,
        0.4, 0.9, 0.8, 0.3, 0.4
    ],
    "face_count": [
        1, 0, 1, 0, 0, 1, 3, 2, 3, 2,
        3, 2, 0, 1, 0, 0, 3, 2, 0, 1
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1, 0, 0
    ]
}

df = pd.DataFrame(data)

# Separate inputs and target
X = df[["text", "brightness", "edge_density", "face_count"]]
y = df["label"]  # 0 = Real, 1 = Fake

# Preprocess text and image-related features
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(), "text"),
    ("image", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["brightness", "edge_density", "face_count"])
])

# Create multimodal classification pipeline
model = Pipeline([
    ("features", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, labels=[0, 1],
    target_names=["Real News", "Fake News"],
    zero_division=0
))

# Predict a new news item
new_news = pd.DataFrame([{
    "text": "Secret miracle cure promises instant recovery",
    "brightness": 0.9,
    "edge_density": 0.8,
    "face_count": 2
}])

prediction = model.predict(new_news)[0]
print("Prediction:",
      "Fake News" if prediction == 1 else "Real News")
