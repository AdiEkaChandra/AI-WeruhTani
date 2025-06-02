from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import random
import pickle
import difflib
import os
import re
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================ Konfigurasi API ============================
API_KEY = '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'  # ← Isi API key OpenRouter kamu di sini
MODEL_ID = 'deepseek/deepseek-r1-0528-qwen3-8b:free'

# ============================ Topik Pertanian ============================
allowed_patterns = [
    r'\bpertanian\b', r'\bilmu pertanian\b', r'\bteknologi pertanian\b',
    r'\bekonomi pertanian\b', r'\bagriculture\b', r'\bfarming\b',
    r'\btani\b', r'\bpadi\b', r'\bgabah\b', r'\bpupuk\b',
    r'\bsawah\b', r'\btanam\b', r'\bpetani\b', r'\bhasil panen\b'
]

def is_allowed_topic(user_input):
    return any(re.search(pattern, user_input.lower()) for pattern in allowed_patterns)

# ============================ Fungsi Chatbot ============================
MAX_LENGTH = 500

def ask_chatbot(question):
    if len(question) > MAX_LENGTH:
        question = question[:MAX_LENGTH]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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

# ============================ Fungsi Prediksi Pengeringan ============================
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
        data.append([berat_awal, jenis_gabah, 25, 14, 42.5, 0.75, 75, 10, "kontinu", lama_pengeringan])
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
        pickle.dump((model, X.columns.tolist()), f, protocol=pickle.HIGHEST_PROTOCOL)

def cari_jenis_gabah_terdekat(input_gabah):
    return difflib.get_close_matches(input_gabah, jenis_gabah_list, n=1, cutoff=0.6)[0] if difflib.get_close_matches(input_gabah, jenis_gabah_list, n=1, cutoff=0.6) else None

def prediksi_waktu_pengeringan(berat_awal, jenis_gabah):
    with open("model_prediksi_pengeringan.pkl", "rb") as f:
        model, feature_names = pickle.load(f)

    if jenis_gabah not in jenis_gabah_list:
        gabah_terdekat = cari_jenis_gabah_terdekat(jenis_gabah)
        if gabah_terdekat:
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

# ============================ Endpoint Flask ============================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400
    if not is_allowed_topic(question):
        return jsonify({"error": "Pertanyaan di luar topik pertanian tidak diperbolehkan."}), 400
    jawaban = ask_chatbot(question)
    return jsonify({"jawaban": jawaban})

@app.route("/prediksi", methods=["POST"])
def prediksi():
    data = request.json
    try:
        berat = int(data.get("berat"))
        jenis = data.get("jenis")
        jam, menit = prediksi_waktu_pengeringan(berat, jenis)
        return jsonify({
            "jenis": jenis,
            "berat": berat,
            "durasi": f"{jam} jam {menit} menit",
            "jam": jam,
            "menit": menit
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============================ Jalankan Flask ============================

if __name__ == "__main__":
    if not os.path.exists("model_prediksi_pengeringan.pkl"):
        print("🔧 Melatih model karena file belum ada...")
        train_model()
    app.run(host="0.0.0.0", port=5000)
