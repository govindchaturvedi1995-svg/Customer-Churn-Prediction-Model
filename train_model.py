import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, models


DATA_PATH = "Telco_Customer_Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]      # <-- columns to drop (identifiers, not predictive)

# ----------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(df.head())

# Drop ID columns if present
for col in ID_COLUMNS:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ----------------------------------------------------------------------
# 3. CLEANING
# ----------------------------------------------------------------------
# TotalCharges is sometimes stored as string with blank spaces in the
# Telco dataset -> convert to numeric, coerce errors to NaN, then fill.
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill numeric NaNs with median, categorical NaNs with mode
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode target column (Yes/No -> 1/0)
target_encoder = LabelEncoder()
df[TARGET_COLUMN] = target_encoder.fit_transform(df[TARGET_COLUMN])

# ----------------------------------------------------------------------
# 4. SEPARATE FEATURES / TARGET
# ----------------------------------------------------------------------
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# ----------------------------------------------------------------------
# 5. ENCODE CATEGORICAL COLUMNS (Label Encoding, one encoder per column
#    so the Streamlit app can reuse them for single-row predictions)
# ----------------------------------------------------------------------
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ----------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 7. SCALE NUMERIC FEATURES
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================================
# MODEL 1: RANDOM FOREST (Scikit-learn)
# ==========================================================================
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Random Forest -> Accuracy: {rf_acc:.4f} | ROC-AUC: {rf_auc:.4f}")
print(classification_report(y_test, rf_preds))

# ==========================================================================
# MODEL 2: DEEP LEARNING ANN (TensorFlow / Keras)
# ==========================================================================
print("\nTraining Deep Learning ANN...")

ann_model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)

history = ann_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

ann_probs = ann_model.predict(X_test_scaled).ravel()
ann_preds = (ann_probs >= 0.5).astype(int)

ann_acc = accuracy_score(y_test, ann_preds)
ann_auc = roc_auc_score(y_test, ann_probs)

print(f"\nANN -> Accuracy: {ann_acc:.4f} | ROC-AUC: {ann_auc:.4f}")
print(classification_report(y_test, ann_preds))

# ==========================================================================
# 8. PICK THE BEST MODEL (based on ROC-AUC)
# ==========================================================================
if rf_auc >= ann_auc:
    best_model_type = "random_forest"
    print(f"\n>>> Best model: Random Forest (AUC {rf_auc:.4f} vs ANN {ann_auc:.4f})")
    joblib.dump(rf_model, "model_rf.pkl")
else:
    best_model_type = "ann"
    print(f"\n>>> Best model: ANN (AUC {ann_auc:.4f} vs RF {rf_auc:.4f})")
    ann_model.save("model_ann.keras")


joblib.dump(rf_model, "model_rf.pkl")
ann_model.save("model_ann.keras")

# ----------------------------------------------------------------------
# 9. SAVE PREPROCESSING OBJECTS + METADATA
# ----------------------------------------------------------------------
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

metadata = {
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "feature_order": X.columns.tolist(),
    "best_model_type": best_model_type,
    "rf_accuracy": rf_acc,
    "rf_auc": rf_auc,
    "ann_accuracy": ann_acc,
    "ann_auc": ann_auc,
    "category_values": {col: df[col].unique().tolist() if col in categorical_cols else None
                         for col in categorical_cols},
}
# Save raw category display values (before encoding) for the Streamlit form
raw_df = pd.read_csv(DATA_PATH)
category_display_values = {
    col: sorted(raw_df[col].dropna().unique().tolist()) for col in categorical_cols
}
metadata["category_display_values"] = category_display_values
metadata["numeric_ranges"] = {
    col: (float(X[col].min()), float(X[col].max()), float(X[col].median()))
    for col in numeric_cols
}

joblib.dump(metadata, "metadata.pkl")

print("\nAll models & preprocessing objects saved successfully!")
print("Files created: model_rf.pkl, model_ann.keras, scaler.pkl, encoders.pkl,")
print("               target_encoder.pkl, metadata.pkl")
print("\nYou can now run:  streamlit run app.py")
