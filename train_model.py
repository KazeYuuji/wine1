import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

# 1. Muat Dataset
# Pastikan file WineQT.csv berada di folder yang sama dengan script ini
try:
    df = pd.read_csv('WineQT.csv')
except FileNotFoundError:
    print("Error: File 'WineQT.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    exit()

# 2. Pra-pemrosesan Data
# Pisahkan fitur (X) dan target (y)
# Kolom 'Id' tidak relevan untuk prediksi, jadi kita hapus
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

# Simpan nama-nama fitur untuk digunakan di Flask
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.joblib')

# 3. Bagi Data menjadi Data Latih dan Data Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Skalakan Fitur
# Naive Bayes tidak secara ketat memerlukan scaling, tetapi bisa membantu performa dan
# merupakan praktik yang baik untuk konsistensi.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Latih Model Naive Bayes
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Model Training Selesai.")
print(f"Akurasi Model: {accuracy:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 7. Simpan Model dan Scaler
# Simpan model dan scaler ke file menggunakan joblib
joblib.dump(model, 'naive_bayes_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\nModel dan Scaler telah disimpan.")
print("File yang dihasilkan: naive_bayes_model.joblib, scaler.joblib, feature_names.joblib")
