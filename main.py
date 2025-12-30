import os
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
import joblib

# =========================
# 1) CSV DOSYALARINI BUL
# =========================
csv_files = sorted(glob.glob("temiz_veri_v*.csv"))
if not csv_files:
    raise FileNotFoundError("temiz_veri_v*.csv bulunamadı")

print(f"{len(csv_files)} adet veri dosyası bulundu:")
for f in csv_files:
    print(" -", f)

# =========================
# 2) BİRLEŞTİR
# =========================
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print("\nToplam veri sayısı:", len(df))

# =========================
# 3) KOLON KONTROLÜ
# =========================
required_cols = ["Voltage", "Current_Import", "Power_Import", "SoC", "Label"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolonlar: {missing}")

# =========================
# 4) NA TEMİZLİĞİ (KRİTİK)
# =========================
# SADECE Label NaN ise at
df = df.dropna(subset=["Label"])

# Sayısal kolonları doldur
num_cols = ["Voltage", "Current_Import", "Power_Import", "SoC"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df["Label"] = df["Label"].astype(int)

# =========================
# 5) FEATURE ENGINEERING
# =========================
df["Power_Ratio"] = df["Power_Import"] / (df["Voltage"] * df["Current_Import"] + 1e-6)
df["SoC_Delta"] = df["SoC"].diff().fillna(0)
df["Current_to_Voltage"] = df["Current_Import"] / (df["Voltage"] + 1e-6)
df["Power_per_SoC"] = df["Power_Import"] / (df["SoC"] + 1e-6)

feature_cols = [
    "Voltage",
    "Current_Import",
    "Power_Import",
    "SoC",
    "Power_Ratio",
    "SoC_Delta",
    "Current_to_Voltage",
    "Power_per_SoC"
]

X = df[feature_cols]
y = df["Label"]

print("\nDEBUG ----------------")
print("X shape:", X.shape)
print("Label dağılımı:")
print(y.value_counts())
print("----------------------")

if X.shape[0] == 0:
    raise ValueError("Tüm satırlar filtrelendi. Veri üretiminde hata var.")

# =========================
# 6) SCALE + SPLIT
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7) MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# =========================
# 8) THRESHOLD (RECALL ODAKLI)
# =========================
THRESHOLD = 0.30
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# =========================
# 9) METRİKLER
# =========================
print("\nMODEL SONUÇLARI")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 10) FEATURE IMPORTANCE
# =========================
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)


