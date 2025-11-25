from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model, scaler, dan nama fitur yang telah dilatih
try:
    model = joblib.load('naive_bayes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
except FileNotFoundError:
    print("Error: File model (.joblib) tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu.")
    # Keluar jika model tidak ditemukan agar tidak error saat di-deploy
    exit()


# Route untuk halaman utama (menampilkan form dan hasil)
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # 1. Ambil data dari form
            input_data = {feature: float(request.form.get(feature)) for feature in feature_names}
            
            # 2. Buat DataFrame dari input
            input_df = pd.DataFrame([input_data])

            # 3. Lakukan scaling pada data input (menggunakan scaler yang sama dengan data latih)
            input_scaled = scaler.transform(input_df)

            # 4. Lakukan prediksi
            predicted_class = model.predict(input_scaled)
            prediction = predicted_class[0]

        except (ValueError, TypeError):
            # Tangani error jika input tidak valid (misalnya, bukan angka)
            prediction = "Error: Mohon masukkan nilai numerik yang valid untuk semua field."

    # Render template HTML, kirim hasil prediksi (jika ada)
    return render_template('index.html', features=feature_names, prediction=prediction)

if __name__ == '__main__':
    # Untuk development lokal
    app.run(debug=True)
