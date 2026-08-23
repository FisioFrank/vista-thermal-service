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
import os
import numpy as np
import flyr
from flask import Flask, request, jsonify
from flask_cors import CORS
from anthropic import Anthropic

app = Flask(__name__)
# En producción, reemplaza "*" por el dominio real de tu app (ej. tu URL de Vercel)
CORS(app, resources={r"/*": {"origins": "*"}})

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
REPORT_MODEL = "claude-sonnet-5"


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


# ============================================================
# Generación de informes con IA — análisis interpretativo por
# módulo, y planes de intervención/entrenamiento. El objetivo:
# que el profesional (CT, PF, médico, fisioterapeuta) vea la
# INTERPRETACIÓN, no los números crudos.
# ============================================================

MODULE_EVIDENCE = {
    "jump": """
Eres un especialista en ciencias del deporte interpretando datos de ForceDecks (CMJ y Drop Jump).
Base de evidencia a aplicar:
- Altura de salto CMJ: una caída sostenida >10% respecto a la línea base individual del atleta
  se asocia con fatiga neuromuscular acumulada o riesgo de sobreentrenamiento (Gathercole et al., 2015).
- RSI-modified (CMJ) y RSI (DJ): reflejan capacidad reactiva/pliométrica; caídas abruptas sin cambio
  de altura sugieren alteración en la estrategia de aterrizaje-despegue, relevante para riesgo de
  lesión de rodilla/tobillo.
- Asimetría de impulso concéntrico y de fuerza de aterrizaje: >10-15% de asimetría interextremidad
  se asocia con mayor riesgo de lesión, especialmente de LCA y isquiotibiales (Bishop et al., 2018;
  Paterno et al., 2010).
- Contexto clínico (Normal / Readaptación-Lesión): una caída de rendimiento en un atleta marcado
  como en readaptación NO debe interpretarse como alarma de lesión nueva, sino como parte esperada
  del proceso — la interpretación debe ajustarse a ese contexto.
""",
    "hrv": """
Eres un especialista en ciencias del deporte interpretando datos de variabilidad de la frecuencia
cardíaca (HRV, medidos con Polar H10 / Kubios).
Base de evidencia a aplicar:
- rMSSD: marcador del tono parasimpático; caídas sostenidas (varios días) respecto a la media móvil
  individual del atleta indican estrés fisiológico acumulado, mala recuperación, o riesgo de
  sobreentrenamiento (Plews et al., 2013; Buchheit, 2014).
- El valor absoluto de rMSSD varía mucho entre individuos — SIEMPRE interpretar respecto a la
  línea base propia del atleta, nunca contra un valor poblacional genérico.
- El índice de Readiness (Kubios) combina rMSSD con otros parámetros; una caída de Readiness sin
  caída proporcional de rMSSD puede reflejar estrés no fisiológico (sueño, estrés psicológico, carga
  externa) y amerita indagar contexto, no solo carga de entrenamiento.
""",
    "gps": """
Eres un especialista en ciencias del deporte interpretando datos de carga externa GPS (Catapult).
Base de evidencia a aplicar:
- Distancia total y distancia a alta velocidad (HSR): picos agudos de carga muy por encima de la
  carga crónica (ratio agudo:crónico >1.5) se asocian con mayor riesgo de lesión de tejido blando
  (Gabbett, 2016).
- Caídas abruptas de velocidad máxima o de distancia HSR respecto al patrón habitual del atleta
  pueden reflejar fatiga, dolor no reportado, o riesgo de lesión muscular incipiente.
""",
    "force": """
Eres un especialista en ciencias del deporte interpretando datos de fuerza (dinamometría).
Base de evidencia a aplicar:
- Asimetrías de fuerza entre extremidades >10-15% son un marcador de riesgo de lesión bien
  establecido, particularmente en tren inferior (Bishop et al., 2018).
- La tasa de desarrollo de fuerza (RFD) es sensible a fatiga neuromuscular incluso cuando la fuerza
  pico se mantiene — vale la pena señalar si RFD cae más que la fuerza máxima.
""",
    "thermal": """
Eres un especialista interpretando termografía infrarroja bilateral en deportistas.
Base de evidencia a aplicar:
- Asimetrías térmicas bilaterales >0.5-1°C en tejido blando pueden reflejar procesos inflamatorios
  o vasculares locales — se usa como screening, nunca como diagnóstico aislado (nunca uses la
  palabra "diagnóstico").
- Los falsos positivos son comunes (variables ambientales, actividad reciente, hidratación de la
  piel) — menciónalo solo si cambia la recomendación concreta (ej. "repetir en condiciones
  controladas"), no como nota académica aparte.
- Una asimetría térmica sostenida en la MISMA zona a través de varias sesiones es más relevante
  que un hallazgo aislado — si los datos no alcanzan para saber si es sostenido o puntual, dilo
  brevemente y pasa directo a la acción a seguir.
- No dediques párrafos a explicar qué tan sólida es la evidencia de la termografía en general;
  concéntrate en qué significa ESTE hallazgo específico y qué hacer con él.
""",
    "overall": """
Eres el especialista que integra TODOS los módulos (ForceDecks, HRV, GPS, Dinamometría, Termografía)
en una lectura única del estado del atleta. Este es el informe más importante del sistema — el que
define si el atleta entrena con normalidad, con ajustes, o necesita intervención.

Estructura tu análisis en DOS bloques claros y explícitos, en este orden:

1. RENDIMIENTO DEPORTIVO — ¿está el atleta en condiciones de rendir hoy? Usa capacidad
   neuromuscular (CMJ/DJ), carga externa (GPS) y fuerza (Dynamo) como las señales que informan esto.

2. RIESGO DE LESIÓN — ¿hay alguna señal, sola o en combinación, que eleve el riesgo? Este es el
   valor real de tener varios módulos juntos: una señal aislada (ej. una asimetría térmica sola)
   pesa poco; la MISMA señal repetida en varios sistemas a la vez (ej. asimetría de aterrizaje en
   ForceDecks + asimetría térmica en la misma pierna + HRV bajo ese día) es una convergencia que
   pesa mucho más que cualquiera de los tres por separado. Busca activamente esas convergencias
   entre módulos — es lo que un profesional no puede ver mirando cada pantalla por separado.

Si un módulo no tiene datos de la sesión de hoy, dilo en una frase y sigue — no rellenes.
""",
}

REPORT_SYSTEM_PROMPT = """Eres un asistente clínico-deportivo que redacta informes para un equipo
profesional (cuerpo técnico, preparador físico, médico deportivo, fisioterapeuta) dentro de VIXTA,
una plataforma de monitoreo de rendimiento y riesgo de lesión.

Reglas estrictas:
1. El profesional que lee esto NO quiere ver los números otra vez — ya los tiene en pantalla. Quiere
   la INTERPRETACIÓN: qué significa, por qué importa, y qué tan urgente es.
2. Cada afirmación relevante debe tener una base en evidencia científica (ya te doy la evidencia
   aplicable abajo) — nunca inventes un umbral o cifra que no te haya dado.
3. Nunca uses la palabra "diagnóstico" — esto es monitoreo y screening, no diagnóstico clínico.
   Si algo amerita evaluación médica, dilo explícitamente ("se recomienda valoración médica"),
   pero no diagnostiques tú.
4. Tono: profesional, directo, sin relleno. Un profesional ocupado debe poder leer esto en 30-45
   segundos y saber qué hacer.
5. Responde en español, sin encabezados markdown tipo "##" — usa párrafos cortos y, si ayuda,
   una lista breve al final con las acciones recomendadas.
6. Si los datos no alcanzan para una conclusión firme, dilo — no rellenes con generalidades vagas.
7. Nunca dediques espacio a explicar qué tan fuerte o débil es la evidencia científica de una
   modalidad en general (eso ya lo sabe el profesional). Ve directo a qué significa ESTE hallazgo
   y qué hacer con él.
8. Estás analizando LA SESIÓN DE HOY (los datos que te doy son del test más reciente, no un
   histórico completo) — habla de "en esta sesión" / "hoy", no generalices sobre "la tendencia del
   atleta" salvo que te haya dado explícitamente datos de comparación con una línea base.
"""


def call_claude_report(system_extra, user_prompt, max_tokens=2000):
    message = anthropic_client.messages.create(
        model=REPORT_MODEL,
        max_tokens=max_tokens,
        system=REPORT_SYSTEM_PROMPT + "\n\n" + system_extra,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text.strip():
        # Diagnóstico: si vuelve vacío, dejamos ver por qué (stop_reason, tipos de bloque)
        # en vez de fallar en silencio otra vez.
        block_types = [b.type for b in message.content]
        raise RuntimeError(f"Respuesta vacía del modelo — stop_reason={message.stop_reason}, bloques={block_types}")
    return text


@app.route("/report", methods=["POST"])
def report():
    """
    Espera JSON:
      {
        "reportType": "module_analysis" | "intervention_plan",
        "module": "jump" | "hrv" | "gps" | "force" | "thermal" | "overall",
        "athleteName": "Nombre del atleta",
        "data": { ...datos ya calculados en el frontend, específicos del módulo... }
      }

    Devuelve: { "text": "..." }
    """
    body = request.get_json(silent=True) or {}
    report_type = body.get("reportType")
    module = body.get("module")
    athlete_name = body.get("athleteName", "el atleta")
    data = body.get("data", {})

    if report_type not in ("module_analysis", "intervention_plan"):
        return jsonify({"error": "reportType debe ser 'module_analysis' o 'intervention_plan'"}), 400
    if module not in MODULE_EVIDENCE:
        return jsonify({"error": f"módulo no reconocido: {module}"}), 400
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "Falta configurar ANTHROPIC_API_KEY en el servidor"}), 500

    evidence = MODULE_EVIDENCE[module]

    if report_type == "module_analysis":
        user_prompt = (
            f"Atleta: {athlete_name}\n"
            f"Módulo: {module}\n"
            f"Datos calculados (JSON):\n{data}\n\n"
            "Redacta el informe interpretativo de este módulo para el equipo profesional, siguiendo "
            "las reglas del sistema. 3-5 párrafos cortos como máximo."
        )
    else:
        user_prompt = (
            f"Atleta: {athlete_name}\n"
            f"Módulo o vista: {module}\n"
            f"Datos calculados (JSON):\n{data}\n\n"
            "Basándote en estos hallazgos, redacta un plan de intervención/entrenamiento concreto y "
            "accionable: qué ajustar en la carga de entrenamiento, qué trabajar en readaptación o "
            "prevención, con qué frecuencia volver a monitorear, y si amerita derivar a valoración "
            "médica. Cierra con una lista breve de 3-6 acciones concretas, priorizadas."
        )

    try:
        text = call_claude_report(evidence, user_prompt, max_tokens=2800 if module == "overall" else 2000)
    except Exception as e:
        return jsonify({"error": f"No se pudo generar el informe: {e}"}), 502

    if not text or not text.strip():
        return jsonify({"error": "El modelo no devolvió texto (respuesta vacía) — intenta de nuevo"}), 502

    return jsonify({"text": text})


if __name__ == "__main__":
    # Solo para pruebas locales. En producción, Render usa gunicorn (ver Procfile / start command).
    app.run(host="0.0.0.0", port=5000, debug=True)
