import pandas as pd
import numpy as np
import random
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import difflib

# Daftar jenis gabah
jenis_gabah_list = [
    "IR64", "Ciherang", "Mekongga", "Inpari 32", "Inpari 42 Agritan GSR",
    "Sintanur", "Cigeulis", "Inpari 13", "Inpari 24", "Inpari 30"
]

# Koefisien waktu per jenis gabah (menit per kg)
koefisien_waktu = {
    "IR64": 0.045,
    "Ciherang": 0.047,
    "Mekongga": 0.046,
    "Inpari 32": 0.048,
    "Inpari 42 Agritan GSR": 0.049,
    "Sintanur": 0.046,
    "Cigeulis": 0.047,
    "Inpari 13": 0.045,
    "Inpari 24": 0.048,
    "Inpari 30": 0.047
}

# Membuat data dummy
def generate_data(n_samples=300):
    data = []
    for _ in range(n_samples):
        berat_awal = random.randint(500, 10000)  # Berat antara 500 - 10000 kg
        jenis_gabah = random.choice(jenis_gabah_list)
        kadar_air_awal = 25
        target_kadar_air = 14
        suhu = 42.5
        kecepatan_udara = 0.75
        kelembaban = 75
        kapasitas_mesin = 10
        metode = "kontinu"

        # Hitung lama pengeringan
        base_time_per_kg = koefisien_waktu[jenis_gabah]
        noise = np.random.uniform(-0.05, 0.05)  # ±5% variasi
        adjusted_time_per_kg = base_time_per_kg * (1 + noise)
        lama_pengeringan = berat_awal * adjusted_time_per_kg
        lama_pengeringan = int(lama_pengeringan)  # Dibulatkan ke menit

        data.append([
            berat_awal, jenis_gabah, kadar_air_awal, target_kadar_air, suhu,
            kecepatan_udara, kelembaban, kapasitas_mesin, metode, lama_pengeringan
        ])

    columns = [
        "berat_awal", "jenis_gabah", "kadar_air_awal", "target_kadar_air",
        "suhu", "kecepatan_udara", "kelembaban", "kapasitas_mesin",
        "metode", "lama_pengeringan"
    ]
    return pd.DataFrame(data, columns=columns)

# Generate dataset
df = generate_data(300)

# Simpan dataset ke CSV
df.to_csv("dataset_pengeringan.csv", index=False)

# Encode jenis gabah dan metode
df_encoded = pd.get_dummies(df, columns=["jenis_gabah", "metode"])

# Split dataset
X = df_encoded.drop("lama_pengeringan", axis=1)
y = df_encoded["lama_pengeringan"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
with open("model_prediksi_pengeringan.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

# Fungsi untuk cari jenis gabah terdekat
def cari_jenis_gabah_terdekat(input_gabah, daftar_gabah):
    hasil = difflib.get_close_matches(input_gabah, daftar_gabah, n=1, cutoff=0.6)
    if hasil:
        return hasil[0]
    else:
        return None

# Fungsi prediksi
def prediksi_waktu_pengeringan(berat_awal, jenis_gabah):
    with open("model_prediksi_pengeringan.pkl", "rb") as f:
        model, feature_names = pickle.load(f)

    jenis_gabah_valid = jenis_gabah_list

    if jenis_gabah not in jenis_gabah_valid:
        gabah_terdekat = cari_jenis_gabah_terdekat(jenis_gabah, jenis_gabah_valid)
        if gabah_terdekat:
            print(f"🔍 Mengoreksi jenis gabah '{jenis_gabah}' menjadi '{gabah_terdekat}'.")
            jenis_gabah = gabah_terdekat
        else:
            raise ValueError(f"Jenis gabah '{jenis_gabah}' tidak dikenali dan tidak ditemukan kemiripan.")

    input_dict = {name: 0 for name in feature_names}
    input_dict['berat_awal'] = berat_awal
    input_dict['kadar_air_awal'] = 25
    input_dict['target_kadar_air'] = 14
    input_dict['suhu'] = 42.5
    input_dict['kecepatan_udara'] = 0.75
    input_dict['kelembaban'] = 75
    input_dict['kapasitas_mesin'] = 10

    gabah_key = f"jenis_gabah_{jenis_gabah}"
    if gabah_key in input_dict:
        input_dict[gabah_key] = 1

    if "metode_kontinu" in input_dict:
        input_dict["metode_kontinu"] = 1

    input_df = pd.DataFrame([input_dict])

    prediksi_menit = model.predict(input_df)[0]
    jam = int(prediksi_menit) // 60
    menit = int(prediksi_menit) % 60

    return jam, menit

# Main program
if __name__ == "__main__":
    berat = int(input("Masukkan berat awal gabah (kg): "))
    jenis = input("Masukkan jenis gabah: ")

    try:
        jam, menit = prediksi_waktu_pengeringan(berat, jenis)
        print(f"✅ Prediksi waktu pengeringan: {jam} jam {menit} menit.")
    except ValueError as e:
        print(f"❌ Error: {e}")
