import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# =========================
# ❤️ HEART DATASET
# =========================
df = pd.read_csv("heart_disease_uci.csv")

df.drop('id', axis=1, inplace=True)
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop('num', axis=1)
y = df['num'].apply(lambda x: 1 if x > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVC(probability=True)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("✅ Heart model done")

# =========================
# 🧪 DIAGNOSIS DATASET
# =========================
df2 = pd.read_csv("medical_Diagnosis_dataset.csv")

df2 = df2.dropna()
df2 = pd.get_dummies(df2, drop_first=True)

target = df2.columns[-1]

X2 = df2.drop(target, axis=1)
y2 = df2[target]

if y2.dtype == 'object':
    y2 = y2.astype('category').cat.codes

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X2_train = scaler2.fit_transform(X2_train)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X2_train, y2_train)

pickle.dump(model2, open("lr_diag.pkl", "wb"))
pickle.dump(scaler2, open("scaler_diag.pkl", "wb"))
pickle.dump(X2.columns.tolist(), open("columns_diag.pkl", "wb"))

print("✅ Diagnosis model done")