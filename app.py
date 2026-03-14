import os
import json
import anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://nedretkademli.github.io", "http://localhost:*", "http://127.0.0.1:*", "null"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def build_prompt(data: dict) -> str:
    current_tools = ", ".join(data.get("current_tools", [])) or "belirtilmemiş"
    ai_goals = ", ".join(data.get("ai_goals", [])) or "belirtilmemiş"
    challenges = ", ".join(data.get("challenges", [])) or "belirtilmemiş"

    return f"""Aşağıdaki bilgilere sahip bir şirket için Türkçe AI olgunluk raporu hazırla.

ŞİRKET BİLGİLERİ:
- Ad: {data.get('name', '')}
- Şirket: {data.get('company', '')}
- Sektör: {data.get('sector', '')}
- Ekip Büyüklüğü: {data.get('team_size', '')}
- AI Farkındalık Seviyesi (1-5): {data.get('ai_awareness', '')}
- Mevcut Kullandığı Dijital Araçlar: {current_tools}
- AI Hedefleri: {ai_goals}
- Başlıca Zorluklar: {challenges}
- Yatırım Hazırlığı: {data.get('budget_ready', '')}
- Anket Skoru: {data.get('score', 0)}/100

GÖREV:
Bu şirkete özel, kişiselleştirilmiş bir AI olgunluk analizi yap.

Aşağıdaki JSON yapısını AYNEN döndür, başka hiçbir şey yazma:

{{
  "level": "<Başlangıç | Gelişen | İleri | Lider>",
  "score": <0-100 arası sayı>,
  "summary": "<{data.get('name', '')} ve {data.get('company', '')} şirketine özel 2-3 cümlelik özet. Kişisel hitap et, güçlü ve geliştirilmesi gereken yönleri belirt.>",
  "dimensions": {{
    "awareness": <0-100 arası sayı, AI farkındalık skoru>,
    "strategy": <0-100 arası sayı, AI strateji hazırlığı skoru>,
    "tools": <0-100 arası sayı, mevcut araç/altyapı skoru>,
    "investment": <0-100 arası sayı, yatırım hazırlığı skoru>
  }},
  "recommendations": [
    "<{data.get('sector', '')} sektörüne ve bu şirketin durumuna özel somut öneri 1>",
    "<Özellikle zayıf boyuta yönelik somut öneri 2>",
    "<Kısa vadede uygulanabilir somut öneri 3>"
  ],
  "workshop_agenda": [
    {{"time": "09:00-10:30", "topic": "<konu başlığı>", "description": "<bu şirketin durumuna göre özelleştirilmiş içerik açıklaması>"}},
    {{"time": "10:30-10:45", "topic": "Kahve Molası", "description": ""}},
    {{"time": "10:45-12:30", "topic": "<konu başlığı>", "description": "<açıklama>"}},
    {{"time": "12:30-13:30", "topic": "Öğle Arası", "description": ""}},
    {{"time": "13:30-15:00", "topic": "<konu başlığı>", "description": "<açıklama>"}},
    {{"time": "15:00-15:15", "topic": "Kahve Molası", "description": ""}},
    {{"time": "15:15-16:30", "topic": "<konu başlığı>", "description": "<açıklama>"}},
    {{"time": "16:30-17:00", "topic": "Soru-Cevap ve Değerlendirme", "description": "Günün özeti, sorular ve katılım sertifikası"}}
  ]
}}

Skoru anket puanı ile tutarlı tut. Seviye belirleme:
- 0-30: Başlangıç
- 31-55: Gelişen
- 56-75: İleri
- 76-100: Lider

Boyut skorlarını şirketin gerçek durumunu yansıtacak şekilde hesapla.
Workshop içeriğini bu şirketin zayıf olduğu alanlara odakla.
Sadece JSON döndür, ek açıklama yazma."""


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Veri bulunamadı"}), 400

    required_fields = ["name", "company", "email"]
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"Zorunlu alan eksik: {field}"}), 400

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system="Sen bir yapay zeka danışmanlık uzmanısın. Şirketler için kişiselleştirilmiş AI olgunluk raporları hazırlıyorsun. Her zaman geçerli JSON formatında yanıt veriyorsun.",
            messages=[
                {"role": "user", "content": build_prompt(data)}
            ]
        )

        text = response.content[0].text.strip()

        # JSON bloğu varsa temizle
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        report = json.loads(text)

        # Skoru ve seviyeyi anket sonucuyla doğrula
        if "score" not in report:
            report["score"] = data.get("score", 50)

        return jsonify(report)

    except json.JSONDecodeError as e:
        return jsonify({"error": "Rapor oluşturulurken format hatası", "detail": str(e)}), 500
    except anthropic.AuthenticationError:
        return jsonify({"error": "API anahtarı geçersiz"}), 401
    except anthropic.RateLimitError:
        return jsonify({"error": "Çok fazla istek. Lütfen biraz bekleyin."}), 429
    except anthropic.APIError as e:
        return jsonify({"error": "API hatası", "detail": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Sunucu hatası", "detail": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
