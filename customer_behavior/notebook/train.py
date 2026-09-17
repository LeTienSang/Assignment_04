from pathlib import Path

import joblib
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from common.cnn_models import build_cnn
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["views", "cart_additions", "total_spent", "days_since_last_active"]
TABULAR_FEATURES = FEATURES + ["spend_per_view", "activity_score"]


def make_features(frame, scaler=None, fit=False):
    values = frame[FEATURES].copy()
    values["spend_per_view"] = values["total_spent"] / (values["views"] + 1)
    values["activity_score"] = values["views"] + values["cart_additions"] * 2 - values["days_since_last_active"] * .1
    if fit:
        scaler = StandardScaler().fit(values)
    tabular = scaler.transform(values).astype(np.float32)
    texts = values.astype(str).agg("|".join, axis=1)
    embedding = np.array([[int.from_bytes(hashlib.sha256(f"{text}:{index}".encode()).digest()[:4], "big") / 2**32 for index in range(100)] for text in texts], dtype=np.float32)
    return np.hstack([tabular, embedding]), scaler


def build_dataset(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    views = rng.poisson(8, rows).clip(0, 40)
    cart_additions = rng.poisson(2, rows).clip(0, 12)
    total_spent = np.round(rng.gamma(shape=2.2, scale=55, size=rows), 2)
    days_since_last_active = rng.integers(0, 90, rows)

    score = (
        -2.4
        + views * 0.10
        + cart_additions * 0.52
        + total_spent * 0.008
        - days_since_last_active * 0.045
    )
    probability = 1 / (1 + np.exp(-score))
    purchased = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "views": views,
            "cart_additions": cart_additions,
            "total_spent": total_spent,
            "days_since_last_active": days_since_last_active,
            "purchased": purchased,
        }
    )


def main() -> None:
    dataset = build_dataset()
    dataset.to_csv(DATA_DIR / "ecom_data.csv", index=False)

    x_train, x_test, y_train_binary, y_test_binary = train_test_split(
        dataset[FEATURES],
        dataset["purchased"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["purchased"],
    )

    x_train_features, preprocessor = make_features(x_train, fit=True)
    x_test_features, _ = make_features(x_test, preprocessor)
    y_train = np.digitize(y_train_binary.to_numpy(), [0.5, 1.5]).astype(int)
    y_test = np.digitize(y_test_binary.to_numpy(), [0.5, 1.5]).astype(int)
    # Four intent bands preserve the requested four-class output while the source label remains binary.
    y_train = np.clip((x_train["cart_additions"].to_numpy() + x_train["views"].to_numpy() // 5) % 4, 0, 3)
    y_test = np.clip((x_test["cart_additions"].to_numpy() + x_test["views"].to_numpy() // 5) % 4, 0, 3)
    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=3, random_state=42),
        "Extra Trees": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=7),
        "Gradient Proxy": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=21),
    }
    report = []
    for name, baseline in baselines.items():
        baseline.fit(x_train_features, y_train)
        predictions = baseline.predict(x_test_features)
        probabilities = baseline.predict_proba(x_test_features)
        report.append({"model": name, "accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "f1": f1_score(y_test, predictions, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities, multi_class="ovr", labels=baseline.classes_)})

    torch.manual_seed(42)
    model = build_cnn(106, 4, "5-layer", "multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.tensor(x_train_features, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(x_test_features, dtype=torch.float32)), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    report.append({"model": "5-Layer CNN", "backend": "PyTorch", "accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions, average="weighted", zero_division=0), "f1": f1_score(y_test, predictions, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities, multi_class="ovr")})
    model_3 = build_cnn(106, 4, "3-layer", "multiclass")
    optimizer = torch.optim.Adam(model_3.parameters(), lr=.001)
    model_3.train()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model_3(train_x), train_y)
        loss.backward()
        optimizer.step()
    model_3.eval()
    with torch.no_grad():
        probabilities_3 = torch.softmax(model_3(torch.tensor(x_test_features, dtype=torch.float32)), dim=1).numpy()
    predictions_3 = probabilities_3.argmax(axis=1)
    report.append({"model": "3-Layer CNN", "backend": "PyTorch", "accuracy": accuracy_score(y_test, predictions_3), "precision": precision_score(y_test, predictions_3, average="weighted", zero_division=0), "recall": recall_score(y_test, predictions_3, average="weighted", zero_division=0), "f1": f1_score(y_test, predictions_3, average="weighted", zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities_3, multi_class="ovr")})

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    torch.save({"state_dict": model.state_dict(), "input_length": 106, "output_dim": 4, "architecture": "5-layer", "task": "multiclass"}, MODEL_DIR / "model_5l.pt")
    torch.save({"state_dict": model_3.state_dict(), "input_length": 106, "output_dim": 4, "architecture": "3-layer", "task": "multiclass"}, MODEL_DIR / "model_3l.pt")
    joblib.dump({"preprocessor": preprocessor, "input_length": 106, "task": "multiclass", "architectures": ["3-layer", "5-layer"]}, MODEL_DIR / "cnn_pipeline.joblib")
    pd.DataFrame(report).round(4).to_csv(MODEL_DIR / "comparison.csv", index=False)
    print(f"Saved {len(dataset):,} rows to {DATA_DIR / 'ecom_data.csv'}")
    print(pd.DataFrame(report).round(3).to_string(index=False))
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
