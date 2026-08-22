"""
Servicio de decodificación radiométrica para VIXTA.

Recibe una foto FLIR (JPG con datos térmicos crudos embebidos) más una
lista de "cajas" (posiciones de zona, en coordenadas relativas 0–1), y
devuelve la temperatura promedio/mínima/máxima dentro de cada caja.

Esto NO reconoce automáticamente qué es cada zona (eso sería un
proyecto de visión por computador aparte) — lee la temperatura real
dentro de las cajas que tú (o la plantilla guardada) ya definieron.
"""

import io
import numpy as np
import flyr
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# En producción, reemplaza "*" por el dominio real de tu app (ej. tu URL de Vercel)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/extract", methods=["POST"])
def extract():
    """
    Espera (multipart/form-data):
      - image: el archivo JPG de FLIR
      - boxes: JSON con una lista de cajas, ej:
          [{"id": "cadera_izq", "x": 0.30, "y": 0.55, "width": 0.15, "height": 0.10}, ...]
        x, y = esquina superior izquierda de la caja, como fracción del ancho/alto de la imagen (0 a 1).
        width, height = tamaño de la caja, también como fracción (0 a 1).

    Devuelve:
      {
        "width": 480, "height": 640,
        "results": [
          {"id": "cadera_izq", "mean": 32.4, "min": 31.1, "max": 33.8},
          ...
        ]
      }
    """
    if "image" not in request.files:
        return jsonify({"error": "falta el archivo 'image'"}), 400
    if "boxes" not in request.form:
        return jsonify({"error": "falta el campo 'boxes' (JSON)"}), 400

    import json

    try:
        boxes = json.loads(request.form["boxes"])
    except Exception as e:
        return jsonify({"error": f"'boxes' no es JSON válido: {e}"}), 400

    image_bytes = request.files["image"].read()

    try:
        thermogram = flyr.unpack(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"No se pudo leer datos térmicos de esta imagen: {e}"}), 422

    celsius = thermogram.celsius  # matriz numpy de temperaturas, una por píxel
    height, width = celsius.shape

    results = []
    for box in boxes:
        x0 = int(box["x"] * width)
        y0 = int(box["y"] * height)
        x1 = int((box["x"] + box["width"]) * width)
        y1 = int((box["y"] + box["height"]) * height)
        # Aseguramos límites válidos dentro de la imagen
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            results.append({"id": box.get("id"), "error": "caja fuera de rango"})
            continue
        region = celsius[y0:y1, x0:x1]
        results.append(
            {
                "id": box.get("id"),
                "mean": round(float(np.mean(region)), 2),
                "min": round(float(np.min(region)), 2),
                "max": round(float(np.max(region)), 2),
            }
        )

    return jsonify({"width": int(width), "height": int(height), "results": results})


if __name__ == "__main__":
    # Solo para pruebas locales. En producción, Render usa gunicorn (ver Procfile / start command).
    app.run(host="0.0.0.0", port=5000, debug=True)
