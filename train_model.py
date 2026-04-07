import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed - skipping. Run: pip install xgboost")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("F:\Downloads\Agriculture_dataset_with_metadata .csv")

BASE_FEATURES = ['N', 'P', 'K', 'Moisture', 'pH', 'Temperature', 'Humidity']
df[BASE_FEATURES] = (
    df[BASE_FEATURES]
    .apply(pd.to_numeric, errors='coerce')
    .fillna(df[BASE_FEATURES].median())
)

# ─────────────────────────────────────────────────────────────
# 2. DERIVE AGRONOMIC LABELS
# ─────────────────────────────────────────────────────────────
def derive_action(row):
    m  = row['Moisture']
    t  = row['Temperature']
    h  = row['Humidity']
    n  = row['N']
    p  = row['P']
    k  = row['K']
    ph = row['pH']

    if m < 15 or (t > 36 and h < 35 and m < 25):
        return 'Irrigate'
    if n < 25 or p < 15 or k < 30 or ph < 5.5 or ph > 7.5:
        return 'Apply Fertilizer'
    if h > 75 and t > 28 and m > 25:
        return 'Apply Pesticide'
    return 'Monitor'

print("Deriving agronomic action labels from sensor readings...")
df['Action_Label'] = df.apply(derive_action, axis=1)
print("Distribution:")
for action, cnt in df['Action_Label'].value_counts().items():
    print(f"  {action:<22} {cnt:>6}  ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
df['water_stress']   = (df['Temperature'] * (100 - df['Humidity'])) / (df['Moisture'] + 1)
df['nutrient_score'] = df['N'] + df['P'] + df['K']
df['ph_deviation']   = abs(df['pH'] - 6.5)

FEATURES = BASE_FEATURES + ['water_stress', 'nutrient_score', 'ph_deviation']

# ─────────────────────────────────────────────────────────────
# 4. PUMP LABEL
# ─────────────────────────────────────────────────────────────
def pump_label(row):
    if row['Action_Label'] == 'Irrigate':
        return 1
    if row['Moisture'] < 15:
        return 1
    if row['Temperature'] > 36 and row['Humidity'] < 35:
        return 1
    return 0

df['Pump_Status'] = df.apply(pump_label, axis=1)

# ─────────────────────────────────────────────────────────────
# 5. ENCODE & SPLIT
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y      = le.fit_transform(df['Action_Label'])
y_pump = df['Pump_Status'].values
X      = df[FEATURES].values

X_train, X_test, y_train, y_test, yp_train, yp_test = train_test_split(
    X, y, y_pump, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 6. ACTION MODELS
# ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, C=1.0),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    "SVM":                 SVC(probability=True, kernel="rbf", C=2.0),
}
if HAS_XGB:
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )

print("\nTraining & Evaluating Action Models:\n")

best_model      = None
best_acc        = 0
best_model_name = ""
results         = {}

for name, m in models.items():
    m.fit(X_train_sc, y_train)
    preds    = m.predict(X_test_sc)
    test_acc = accuracy_score(y_test, preds)
    cv_acc   = cross_val_score(m, X_train_sc, y_train, cv=3, scoring='accuracy').mean()

    results[name] = {
        "test_accuracy": round(test_acc, 4),
        "cv_accuracy":   round(cv_acc, 4),
    }
    marker = " <- BEST" if test_acc > best_acc else ""
    print(f"  {name:<25}  Test: {test_acc:.4f}  CV: {cv_acc:.4f}{marker}")

    if test_acc > best_acc:
        best_acc        = test_acc
        best_model      = m
        best_model_name = name

print(f"\nBest Model: {best_model_name}  (Test Accuracy: {best_acc:.4f})")
print("\nClassification Report:")
print(classification_report(y_test, best_model.predict(X_test_sc), target_names=le.classes_))

# ─────────────────────────────────────────────────────────────
# 7. PUMP MODEL
# ─────────────────────────────────────────────────────────────
print("Training Pump ON/OFF Model...")
pump_model = RandomForestClassifier(n_estimators=200, random_state=42)
pump_model.fit(X_train_sc, yp_train)
pump_acc = accuracy_score(yp_test, pump_model.predict(X_test_sc))
print(f"  Pump Model Accuracy: {pump_acc:.4f}")

# ─────────────────────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────────────────────
joblib.dump(best_model,      "model.pkl")
joblib.dump(scaler,          "scaler.pkl")
joblib.dump(le,              "label_encoder.pkl")
joblib.dump(best_model_name, "model_name.pkl")
joblib.dump(results,         "model_results.pkl")
joblib.dump(pump_model,      "pump_model.pkl")
joblib.dump(FEATURES,        "features.pkl")

print("\nAll artifacts saved successfully!")
print(f"Best model: {best_model_name} — {best_acc*100:.2f}% accuracy")