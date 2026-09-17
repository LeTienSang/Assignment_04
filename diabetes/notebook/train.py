from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from common.cnn_models import build_cnn
DATA = ROOT / "data/diabetes.csv"
MODEL = ROOT / "model"
MODEL.mkdir(exist_ok=True)
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
    "precision": precision_score(y_true, predictions, zero_division=0),
    "recall": recall_score(y_true, predictions, zero_division=0),
    "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }

def train_torch(x_train, y_train, x_test, architecture):
    torch.manual_seed(42)
    model = build_cnn(x_train.shape[1], 1, architecture, "binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    train_x = torch.tensor(x_train, dtype=torch.float32)
    train_y = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
    loss = loss_fn(model(train_x), train_y)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = model(torch.tensor(x_test, dtype=torch.float32)).numpy().ravel()
    return model, probabilities

def tensorflow_metrics(x_train, y_train, x_test, y_test, architecture):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    tf.random.set_seed(42)
    layers = [tf.keras.layers.Input((x_train.shape[1], 1)), tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu")]
    if architecture == "3-layer":
        layers += [tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")]
    else:
        layers += [tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu"), tf.keras.layers.MaxPooling1D(2), tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")]
    layers += [tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dense(1, activation="sigmoid")]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(x_train[..., None], y_train.to_numpy(), epochs=60, batch_size=32, verbose=0)
    return metrics(y_test, model.predict(x_test[..., None], verbose=0).ravel())

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    df = pd.read_csv(DATA)
    df[FEATURES] = df[FEATURES].replace(0, np.nan)
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df["Outcome"], test_size=0.2, random_state=42, stratify=df["Outcome"])
    prep = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), FEATURES)])
    x_train_scaled = prep.fit_transform(x_train).astype(np.float32)
    x_test_scaled = prep.transform(x_test).astype(np.float32)
    report = []
    for name, baseline in {
    "Logistic Regression": __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }.items():
        baseline.fit(x_train_scaled, y_train)
        report.append({"model": name, "backend": "sklearn", **metrics(y_test, baseline.predict_proba(x_test_scaled)[:, 1])})

    trained = {}
    for architecture in ("3-layer", "5-layer"):
        model, probabilities = train_torch(x_train_scaled, y_train, x_test_scaled, architecture)
        trained[architecture] = model
        report.append({"model": f"{architecture} CNN", "backend": "PyTorch", **metrics(y_test, probabilities)})
        tf_result = tensorflow_metrics(x_train_scaled, y_train, x_test_scaled, y_test, architecture)
        if tf_result:
            report.append({"model": f"{architecture} CNN", "backend": "TensorFlow/Keras", **tf_result})

    joblib.dump({"preprocessor": prep, "input_shape": [1, len(FEATURES)], "task": "binary", "architectures": ["3-layer", "5-layer"]}, MODEL / "cnn_pipeline.joblib")
    for architecture, model in trained.items():
        torch.save({"state_dict": model.state_dict(), "input_length": len(FEATURES), "output_dim": 1, "architecture": architecture, "task": "binary"}, MODEL / f"model_{architecture[0]}l.pt")
    pd.DataFrame(report).round(4).to_csv(MODEL / "comparison.csv", index=False)
    print(pd.DataFrame(report).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
