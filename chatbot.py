import requests

API_KEY = ''  # Ganti dengan API Key OpenRouter kamu

def is_allowed_topic(user_input):
    allowed_keywords = ['pertanian', 'ilmu pertanian', 'teknologi pertanian', 'ekonomi pertanian', 'agriculture', 'farming']
    return any(keyword.lower() in user_input.lower() for keyword in allowed_keywords)

def ask_chatbot(question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "",  # Ganti dengan domain kamu jika perlu, bisa dikosongkan saat lokal
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen/qwen3-0.6b-04-28:free",
        "messages": [
            {"role": "system", "content": "Kamu adalah chatbot ahli dalam pertanian, ilmu pertanian, teknologi pertanian, dan ekonomi pertanian."},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    print("👩‍🌾 Selamat datang di Chatbot Pertanian!")
    print("Ketik 'keluar' untuk mengakhiri.\n")

    while True:
        user_input = input("Anda: ")

        if user_input.lower() in ["keluar", "exit"]:
            print("👋 Terima kasih telah menggunakan Chatbot Pertanian.")
            break

        if is_allowed_topic(user_input):
            jawaban = ask_chatbot(user_input)
            print("🤖 Chatbot:", jawaban)
        else:
            print("⚠️ Maaf, saya hanya menjawab pertanyaan seputar pertanian.")
