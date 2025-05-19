import requests
import pandas as pd
import numpy as np
import random
import pickle
import difflib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ============================ Konfigurasi API ============================
API_KEY = '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'
MODEL_ID = 'mistralai/mixtral-8x7b-instruct'  

# ============================ Topik Pertanian ============================
def is_allowed_topic(user_input):
    allowed_keywords = ['pertanian', 'ilmu pertanian', 'teknologi pertanian', 'ekonomi pertanian', 'agriculture', 'farming']
    return any(keyword.lower() in user_input.lower() for keyword in allowed_keywords)

# ============================ Chatbot Pertanian ============================
def ask_chatbot(question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "",  # Kosongkan untuk lokal
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "Kamu adalah chatbot ahli dalam pertanian, ilmu pertanian, teknologi pertanian, dan ekonomi pertanian."},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# ============================ Prediksi Pengeringan Gabah ============================
jenis_gabah_list = [
    "IR64", "Ciherang", "Mekongga", "Inpari 32", "Inpari 42 Agritan GSR",
    "Sintanur", "Cigeulis", "Inpari 13", "Inpari 24", "Inpari 30"
]

koefisien_waktu = {
    "IR64": 0.045, "Ciherang": 0.047, "Mekongga": 0.046, "Inpari 32": 0.048,
    "Inpari 42 Agritan GSR": 0.049, "Sintanur": 0.046, "Cigeulis": 0.047,
    "Inpari 13": 0.045, "Inpari 24": 0.048, "Inpari 30": 0.047
}

def generate_data(n_samples=300):
    data = []
    for _ in range(n_samples):
        berat_awal = random.randint(500, 10000)
        jenis_gabah = random.choice(jenis_gabah_list)
        lama_pengeringan = int(berat_awal * koefisien_waktu[jenis_gabah] * (1 + np.random.uniform(-0.05, 0.05)))

        data.append([
            berat_awal, jenis_gabah, 25, 14, 42.5, 0.75, 75, 10, "kontinu", lama_pengeringan
        ])

    columns = [
        "berat_awal", "jenis_gabah", "kadar_air_awal", "target_kadar_air",
        "suhu", "kecepatan_udara", "kelembaban", "kapasitas_mesin",
        "metode", "lama_pengeringan"
    ]
    return pd.DataFrame(data, columns=columns)

def train_model():
    df = generate_data(300)
    df.to_csv("dataset_pengeringan.csv", index=False)
    df_encoded = pd.get_dummies(df, columns=["jenis_gabah", "metode"])
    X = df_encoded.drop("lama_pengeringan", axis=1)
    y = df_encoded["lama_pengeringan"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open("model_prediksi_pengeringan.pkl", "wb") as f:
        pickle.dump((model, X.columns.tolist()), f)

def cari_jenis_gabah_terdekat(input_gabah):
    return difflib.get_close_matches(input_gabah, jenis_gabah_list, n=1, cutoff=0.6)[0] if difflib.get_close_matches(input_gabah, jenis_gabah_list, n=1, cutoff=0.6) else None

def prediksi_waktu_pengeringan(berat_awal, jenis_gabah):
    with open("model_prediksi_pengeringan.pkl", "rb") as f:
        model, feature_names = pickle.load(f)

    if jenis_gabah not in jenis_gabah_list:
        gabah_terdekat = cari_jenis_gabah_terdekat(jenis_gabah)
        if gabah_terdekat:
            print(f"🔍 Mengoreksi jenis gabah '{jenis_gabah}' menjadi '{gabah_terdekat}'.")
            jenis_gabah = gabah_terdekat
        else:
            raise ValueError(f"Jenis gabah '{jenis_gabah}' tidak dikenali.")

    input_dict = {name: 0 for name in feature_names}
    input_dict.update({
        'berat_awal': berat_awal,
        'kadar_air_awal': 25,
        'target_kadar_air': 14,
        'suhu': 42.5,
        'kecepatan_udara': 0.75,
        'kelembaban': 75,
        'kapasitas_mesin': 10,
        f'jenis_gabah_{jenis_gabah}': 1,
        'metode_kontinu': 1
    })

    input_df = pd.DataFrame([input_dict])
    prediksi_menit = model.predict(input_df)[0]
    return int(prediksi_menit) // 60, int(prediksi_menit) % 60

# ============================ Program Utama ============================
if __name__ == "__main__":
    try:
        with open("model_prediksi_pengeringan.pkl", "rb"):
            pass
    except FileNotFoundError:
        print("🔧 Model belum dilatih, melatih sekarang...")
        train_model()

    print("👩‍🌾 Selamat datang di Chatbot Pertanian + Prediksi Waktu Pengeringan!")
    print("Ketik 'keluar' untuk mengakhiri.\n")

    while True:
        user_input = input("Anda: ")

        if user_input.lower() in ["keluar", "exit"]:
            print("👋 Terima kasih telah menggunakan sistem ini.")
            break

        elif 'prediksi' in user_input.lower():
            try:
                berat = int(input("⚖️ Masukkan berat awal gabah (kg): "))
                jenis = input("🌾 Masukkan jenis gabah: ")
                jam, menit = prediksi_waktu_pengeringan(berat, jenis)
                print(f"✅ Prediksi waktu pengeringan: {jam} jam {menit} menit.")
            except ValueError as e:
                print(f"❌ Error: {e}")
        elif is_allowed_topic(user_input):
            jawaban = ask_chatbot(user_input)
            print("🤖 Chatbot:", jawaban)
        else:
            print("⚠️ Maaf, saya hanya menjawab pertanyaan seputar pertanian atau melakukan prediksi pengeringan.")