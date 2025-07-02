import matplotlib
matplotlib.use('Agg')
from flask import Flask, jsonify, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
import io
from base64 import b64encode
from flask_cors import CORS
import requests
import pickle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# MODELLERİ YÜKLE
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("xgb_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("regression_model.pkl", "rb") as f:
    litre_model = pickle.load(f)

with open("regression_scaler.pkl", "rb") as f:
    litre_scaler = pickle.load(f)

app = Flask(__name__)
CORS(app)

# VERİ YÜKLE
data = pd.read_excel("veri_seti_1yil.xlsx")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(data.to_dict(orient="records"))

# GRAFİK OLUŞTUR
def create_graph(y_data, y_label, title, color):
    plt.style.use('ggplot')  # Modern ve sade stil
    fig, ax = plt.subplots(figsize=(14, 6))

    # Son 100 veriyi al (gerekirse artır)
    x_data = pd.to_datetime(data["TarihSaat"][-100:])
    y_data = y_data[-100:]

    # Grafik çizimi
    ax.plot(x_data, y_data, label=title, color=color, linewidth=2, marker='o', markersize=4)

    # Başlık ve etiketler
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Tarih - Saat", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # X ekseni formatı ve yoğunluğa göre gösterim
    ax.set_xlim([x_data.min(), x_data.max()])
    if len(x_data) > 60:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    elif len(x_data) > 100:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    fig.autofmt_xdate(rotation=30)

    # Stil detayları
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Görseli base64 olarak döndür
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=150, bbox_inches='tight')
    img.seek(0)
    plt.close()
    return img

@app.route("/graph/all", methods=["GET"])
def graph_all():
    print("Grafikler oluşturuluyor...")

    graphs = {
        "Toprak Nemi": create_graph(data["ToprakNemi(%)"], "Toprak Nemi (%)", "Toprak Nemi Grafiği", "blue"),
        "Hava Sıcaklığı": create_graph(data["HavaSicakligi(°C)"], "Sıcaklık (°C)", "Hava Sıcaklığı Grafiği", "red"),
        "Hava Nemi": create_graph(data["HavaNemi(%)"], "Hava Nemi (%)", "Hava Nemi Grafiği", "green"),
        "Işık Yoğunluğu": create_graph(data["IsikYogunlugu(lux)"], "Işık Yoğunluğu (lux)", "Işık Yoğunluğu Grafiği", "orange")
    }
    encoded_graphs = {
        title: f"data:image/png;base64,{b64encode(img.read()).decode()}"
        for title, img in graphs.items()
    }
    return jsonify(encoded_graphs)

@app.route("/thingspeak", methods=["GET"])
def get_thingspeak_predictions():
    try:
        api_url = "https://api.thingspeak.com/channels/2736785/feeds.json?api_key=RE1I0AGF14BCQZLD&results=10"
        response = requests.get(api_url)
        response.raise_for_status()
        thingspeak_data = response.json()

        formatted_data = []
        for feed in thingspeak_data["feeds"]:
            if all(feed.get(f"field{i}") for i in range(1, 8)):
                formatted_data.append({
                    "time": feed["created_at"],
                    "ToprakNemi(%)": float(feed["field4"]),
                    "HavaSicakligi(°C)": float(feed["field2"]),
                    "HavaNemi(%)": float(feed["field1"]),
                    "IsikYogunlugu(lux)": float(feed["field3"]),
                    "R": float(feed["field5"]) * 255 / 1024,
                    "G": float(feed["field6"]) * 255 / 1024,
                    "B": float(feed["field7"]) * 255 / 1024
                })

        if not formatted_data:
            return jsonify({"status": "error", "message": "Yeterli veri yok."})

        df = pd.DataFrame(formatted_data)
        features = ['ToprakNemi(%)', 'HavaSicakligi(°C)', 'HavaNemi(%)', 'IsikYogunlugu(lux)', 'R', 'G', 'B']
        X = df[features]
        timestamps = df["time"].tolist()

        # Ölçekleme ve tahmin
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Tahminleri JSON olarak paketle
        results = []
        for i in range(len(predictions)):
            results.append({
                "time": timestamps[i],
                "SulamaDurumu": "Sulama Gerekli" if predictions[i] == 1 else "Sulama Gereksiz",
                "Olasilik": float(round(probabilities[i] * 100, 2))
            })

        return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/predict_litre", methods=["POST"])
def predict_litre():
    try:
        content = request.get_json()
        required_keys = ["ToprakNemi(%)", "HavaSicakligi(°C)", "HavaNemi(%)", "IsikYogunlugu(lux)"]
        if not all(key in content for key in required_keys):
            return jsonify({"status": "error", "message": "Eksik veri alanları"}), 400

        test_sample = [[
            content["ToprakNemi(%)"],
            content["HavaSicakligi(°C)"],
            content["HavaNemi(%)"],
            content["IsikYogunlugu(lux)"]
        ]]
        test_sample_scaled = litre_scaler.transform(test_sample)
        predicted_litre = float(litre_model.predict(test_sample_scaled)[0])

        return jsonify({
            "status": "success",
            "TahminiSuMiktari(Litre)": predicted_litre
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
