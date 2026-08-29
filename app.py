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
import json as json_lib
import subprocess
import tempfile
import numpy as np
import pillow_jpls  # registra el códec JPEG-LS en Pillow — necesario para leer el térmico crudo de FLIR
from PIL import Image
import requests
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


@app.route("/debug-exif", methods=["POST"])
def debug_exif():
    """
    Diagnóstico: recibe una foto y devuelve los metadatos EXIF relevantes
    (los que tienen que ver con datos térmicos FLIR), para entender cómo
    esta cámara específica guarda su información antes de intentar decodificarla.
    """
    if "image" not in request.files:
        return jsonify({"error": "falta el archivo 'image'"}), 400

    image_bytes = request.files["image"].read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            ["exiftool", "-j", "-G", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return jsonify({"error": f"exiftool falló: {result.stderr}"}), 500

        data = json_lib.loads(result.stdout)[0]
        relevant_terms = ["Thermal", "Planck", "Emissivity", "Camera", "Raw", "Atmospheric", "Object", "Reflected", "IR", "Model"]
        relevant = {k: v for k, v in data.items() if any(term in k for term in relevant_terms)}
        # Los valores binarios crudos son enormes — los resumimos en vez de mandarlos completos.
        for k, v in list(relevant.items()):
            if isinstance(v, str) and len(v) > 200:
                relevant[k] = f"(binario, {len(v)} caracteres)"

        # Extrae el dato térmico crudo directamente (sin flirimageextractor de por medio)
        # y revisa qué es de verdad, para diagnosticar el formato exacto.
        raw_result = subprocess.run(
            ["exiftool", "-b", "-RawThermalImage", tmp_path],
            capture_output=True,
            timeout=30,
        )
        raw_bytes = raw_result.stdout
        raw_info = {"bytes_extraidos": len(raw_bytes), "primeros_bytes_hex": raw_bytes[:16].hex() if raw_bytes else None}
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw_bytes))
            raw_info["formato_detectado_por_PIL"] = img.format
            raw_info["modo"] = img.mode
            raw_info["tamano"] = img.size
        except Exception as e:
            raw_info["error_al_abrir_con_PIL"] = str(e)

        return jsonify({"total_de_campos": len(data), "campos_relevantes": relevant, "dato_crudo": raw_info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _exiftool_value(tmp_path, tag):
    """Lee un solo tag numérico de exiftool para esta foto (ej. PlanckR1)."""
    result = subprocess.run(["exiftool", f"-{tag}", "-s3", tmp_path], capture_output=True, text=True, timeout=30)
    val = result.stdout.strip()
    if not val:
        raise ValueError(f"La foto no tiene el campo '{tag}' — puede que no sea una foto FLIR radiométrica.")
    # Algunos campos vienen como "20.0 C" — nos quedamos solo con el número.
    num = "".join(c for c in val if c.isdigit() or c in ".-")
    return float(num)


def decode_flir_celsius(tmp_path):
    """
    Extrae la temperatura real de una foto FLIR (probado con FLIR C5, que guarda
    el dato crudo como JPEG-LS — un formato que Pillow no lee sin el plugin
    pillow_jpls). Aplica la fórmula radiométrica de Planck con las constantes de
    calibración propias de CADA foto (no son valores fijos, vienen en sus metadatos).
    """
    raw_result = subprocess.run(["exiftool", "-b", "-RawThermalImage", tmp_path], capture_output=True, timeout=30)
    raw_bytes = raw_result.stdout
    if not raw_bytes:
        raise ValueError("Esta foto no tiene datos térmicos crudos embebidos (RawThermalImage vacío).")

    img = Image.open(io.BytesIO(raw_bytes))
    raw = np.array(img).astype(np.float64)

    planck_r1 = _exiftool_value(tmp_path, "PlanckR1")
    planck_r2 = _exiftool_value(tmp_path, "PlanckR2")
    planck_b = _exiftool_value(tmp_path, "PlanckB")
    planck_f = _exiftool_value(tmp_path, "PlanckF")
    planck_o = _exiftool_value(tmp_path, "PlanckO")
    emissivity = _exiftool_value(tmp_path, "Emissivity")
    reflected_c = _exiftool_value(tmp_path, "ReflectedApparentTemperature")

    reflected_k = reflected_c + 273.15
    raw_refl = planck_r1 / (planck_r2 * (np.exp(planck_b / reflected_k) - planck_f)) - planck_o
    raw_obj = (raw - (1 - emissivity) * raw_refl) / emissivity
    temp_k = planck_b / np.log(planck_r1 / (planck_r2 * (raw_obj + planck_o)) + planck_f)
    return temp_k - 273.15


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

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        celsius = decode_flir_celsius(tmp_path)
    except Exception as e:
        return jsonify({"error": f"No se pudo leer datos térmicos de esta imagen: {e}"}), 422
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
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
        shape = box.get("shape", "rect")
        if shape in ("circle", "ellipse", "oval"):
            # Máscara elíptica inscrita en la caja — solo mide los píxeles
            # que caen dentro de la forma real, no todo el rectángulo.
            h, w = region.shape
            cy, cx = h / 2, w / 2
            ry, rx = max(h / 2, 1e-6), max(w / 2, 1e-6)
            yy, xx = np.ogrid[:h, :w]
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            values = region[mask]
        else:
            values = region
        if values.size == 0:
            results.append({"id": box.get("id"), "error": "zona demasiado pequeña"})
            continue
        results.append(
            {
                "id": box.get("id"),
                "mean": round(float(np.mean(values)), 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
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
Si el dato incluye "esRecordPersonalDeAltura" o "esRecordPersonalDeRSI" en true, dilo explícitamente
al inicio del análisis — es información relevante para el atleta y el cuerpo técnico, no la omitas.
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
- ACWR ("acwr" — ratio de carga aguda de 7 días : carga crónica de 28 días, usando Player Load): el
  marcador de riesgo de lesión por carga mejor establecido en la literatura. >1.5 se asocia a mayor
  riesgo de lesión de tejido blando; <0.8 puede indicar un descenso brusco de carga (relevante para
  destrenamiento, no solo para exceso de carga). Si "acwrHistorialInsuficiente" es true, dilo
  explícitamente y trata el ACWR como una referencia preliminar, no como un dato firme — el cálculo
  necesita ~28 días de historial para ser confiable.
- Desaceleraciones de alta intensidad son mecánicamente más demandantes que las aceleraciones —
  se han vinculado a mayor riesgo de isquiotibiales/rodilla. Compara el conteo de aceleraciones vs
  desaceleraciones: un desbalance marcado (muchas más desaceleraciones) es relevante para riesgo,
  no solo el volumen total.
- Player Load por minuto (intensidad de la sesión) es más informativo que el Player Load total para
  comparar sesiones de duración distinta — una sesión corta muy intensa no es lo mismo que una larga
  y suave, aunque el Player Load total sea similar.
- Distancia total y distancia a alta velocidad (HSR): picos agudos de carga muy por encima de la
  carga crónica se asocian con mayor riesgo de lesión de tejido blando (ver ACWR arriba).
- Caídas abruptas de velocidad máxima o de distancia HSR respecto al patrón habitual del atleta
  pueden reflejar fatiga, dolor no reportado, o riesgo de lesión muscular incipiente.
""",
    "force": """
Eres un especialista en ciencias del deporte interpretando datos de dinamometría manual (fuerza
isométrica de cuádriceps, isquiotibial, glúteo, y rotadores internos/externos de cadera).
Base de evidencia a aplicar:
- Asimetrías de fuerza entre extremidades >10-15% son un marcador de riesgo de lesión bien
  establecido, particularmente en tren inferior.
- La tasa de desarrollo de fuerza (RFD) es sensible a fatiga neuromuscular incluso cuando la fuerza
  pico se mantiene — vale la pena señalar si RFD cae más que la fuerza máxima.
- Asimetría de RFD ("asimetriaRfdPct") es un marcador DISTINTO de la asimetría de fuerza máxima —
  un atleta puede tener fuerza máxima simétrica pero producirla mucho más rápido de un lado, lo cual
  también es relevante para riesgo de lesión y no debe ignorarse solo porque la fuerza máxima esté
  balanceada.
- Ratio Isquiotibial:Cuádriceps ("ratioIsquiotibialCuadriceps_izquierdo_pct" y "_derecho_pct"): el
  marcador de riesgo de lesión de isquiotibiales/LCA mejor establecido en la literatura. Un ratio
  <60% se asocia a mayor riesgo; 60-80% se considera un rango funcional aceptable en la mayoría de
  deportes de campo. SIEMPRE evalúa el ratio POR LADO — el promedio de ambos lados
  ("_promedioAmbosLados_pct") puede ocultar que una pierna está comprometida mientras la otra
  compensa; menciona el promedio solo como referencia secundaria, nunca como el dato principal.
- Ratio Rotador interno:externo de cadera (mismo patrón: "_izquierdo_pct" / "_derecho_pct" /
  "_promedioAmbosLados_pct"): un desbalance marcado se ha vinculado a mayor riesgo de lesión de
  ingle/cadera, especialmente en deportes con cambios de dirección frecuentes. Igual que con H:Q,
  evalúa por lado primero — la evidencia aquí es menos extensa que la de H:Q, así que interpreta con
  más cautela.
- Si no hay datos de ambos grupos musculares el mismo día, estos ratios no estarán disponibles — no
  los inventes.
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
- Cada zona trae su propio "contextoClinico" ("Normal", "Dolor reportado", "Lesión previa",
  "Lesión actual") y a veces una "notaClinica". Esto CAMBIA el peso del hallazgo térmico:
  - Una asimetría térmica baja (incluso por debajo del umbral de Monitoreo) en una zona con dolor
    o lesión reportada es MÁS relevante clínicamente que la misma asimetría en una zona sin
    síntomas — dilo explícitamente, no la trates igual que un hallazgo aislado sin contexto.
  - Si una zona tiene dolor/lesión reportada pero el delta térmico es bajo, no lo descartes por el
    número — la ausencia de asimetría térmica marcada no descarta el problema clínico, solo dice
    que términografía no lo está mostrando hoy.
  - Si hay contexto de lesión actual, la recomendación de valoración médica/fisioterapéutica pesa
    más que si fuera un hallazgo puramente térmico sin síntomas.
""",
    "overall": """
Eres el especialista que integra TODOS los módulos (ForceDecks, HRV, GPS, Dinamometría, Termografía)
en una lectura única del estado del atleta. Este es el informe más importante del sistema — el que
define si el atleta entrena con normalidad, con ajustes, o necesita intervención.

"señalReadiness" es un promedio ponderado de "scorePorModulo" (ForceDecks 25%, HRV 25%, Dynamo 25%,
GPS 15%, Termografía 10% — pesos por solidez de evidencia; un módulo sin datos simplemente no entra
al promedio, no penaliza). Puedes mencionar el número, pero explica SIEMPRE a partir de los módulos
individuales que lo componen, nunca trates el número global como un dato aislado sin desglose.

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

LÍMITE DE EXTENSIÓN — esto es estricto, no una sugerencia: máximo 2 párrafos cortos por bloque
(Rendimiento y Riesgo), de 3-4 líneas cada uno, más una lista final de 3-5 acciones. Si tienes
varios hallazgos, prioriza los 2-3 más relevantes y omite el resto — no intentes cubrir cada
módulo con el mismo nivel de detalle. Un profesional ocupado debe terminar de leerlo en 30-45
segundos; si tu borrador es más largo que eso, recórtalo antes de responder.
""",
}

REPORT_SYSTEM_PROMPT = """Eres un asistente clínico-deportivo que redacta informes para un equipo
profesional (cuerpo técnico, preparador físico, médico deportivo, fisioterapeuta) dentro de VIXTA,
una plataforma de monitoreo de rendimiento y riesgo de lesión.

Reglas estrictas:
1. El profesional que lee esto NO quiere ver los números otra vez — ya los tiene en pantalla. Quiere
   la INTERPRETACIÓN: qué significa, por qué importa, y qué tan urgente es.
2. Cada afirmación relevante debe tener una base en evidencia científica (ya te doy la evidencia
   aplicable abajo) — nunca inventes un umbral o cifra que no te haya dado. PERO nunca cites
   autores ni años en el texto (nada de "(Bishop et al., 2018)") — aplica el conocimiento de forma
   directa, sin formato de cita académica. El profesional confía en el sistema, no necesita ver la
   referencia bibliográfica cada vez.
3. Nunca uses la palabra "diagnóstico" — esto es monitoreo y screening, no diagnóstico clínico.
   Si algo amerita evaluación médica, dilo explícitamente ("se recomienda valoración médica"),
   pero no diagnostiques tú.
4. Tono: profesional, directo, sin relleno. Un profesional ocupado debe poder leer esto en 30-45
   segundos y saber qué hacer.
5. Responde en español, sin encabezados markdown tipo "##" — usa párrafos cortos y, si ayuda,
   una lista breve al final con las acciones recomendadas.
6. Si los datos no alcanzan para una conclusión firme, dilo — no rellenes con generalidades vagas.
   Si ves "esPrimeraEvaluacionDelAtleta": true o "comparacionConLineaBase": null, es la PRIMERA vez
   que se evalúa a este atleta en ese módulo. En ese caso está PROHIBIDO usar las palabras "récord
   personal", "punto más alto histórico", "su mejor marca" o cualquier variante — aunque el dato sea
   técnicamente el valor más alto registrado, decirlo así es engañoso porque no hay nada previo con
   qué compararlo. Tampoco digas "en línea con su promedio reciente" (no existe ese promedio aún).
   Frase correcta a usar en su lugar: "esta es la primera medición registrada de [atleta] en este
   módulo — servirá como referencia para futuras sesiones, aún no hay tendencia que evaluar."
   Un campo "esRecordPersonalDe..." en true SOLO es válido cuando "esPrimeraEvaluacionDelAtleta" es
   false — si ambos aparecen juntos, respeta "esPrimeraEvaluacionDelAtleta" y no menciones récord.
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
        thinking={"type": "disabled"},  # no necesitamos razonamiento extendido para redactar informes —
        # sin esto, el modelo puede gastar todo max_tokens "pensando" y nunca escribir el texto visible.
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
        text = call_claude_report(evidence, user_prompt, max_tokens=3000 if module == "overall" else 2500)
    except Exception as e:
        return jsonify({"error": f"No se pudo generar el informe: {e}"}), 502

    if not text or not text.strip():
        return jsonify({"error": "El modelo no devolvió texto (respuesta vacía) — intenta de nuevo"}), 502

    return jsonify({"text": text})


# ============================================================
# Contacto de soporte — envía un correo real a la bandeja del
# administrador vía Resend, con lo que la persona escribió desde
# VIXTA. No se guarda nada en base de datos, solo se envía.
# ============================================================
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
SUPPORT_EMAIL_TO = os.environ.get("SUPPORT_EMAIL_TO", "frankcastro.fisioterapia@gmail.com")


@app.route("/support-message", methods=["POST"])
def support_message():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    user_email = body.get("userEmail", "desconocido")
    org_name = body.get("orgName", "")

    if not message:
        return jsonify({"error": "El mensaje no puede estar vacío"}), 400
    if not RESEND_API_KEY:
        return jsonify({"error": "Falta configurar RESEND_API_KEY en el servidor"}), 500

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": "VIXTA Soporte <onboarding@resend.dev>",
                "to": [SUPPORT_EMAIL_TO],
                "subject": f"Nuevo mensaje de soporte — {org_name or user_email}",
                "text": f"De: {user_email}\nClub: {org_name or '(sin nombre)'}\n\nMensaje:\n{message}",
            },
            timeout=15,
        )
        if resp.status_code >= 300:
            return jsonify({"error": f"Resend devolvió un error ({resp.status_code}): {resp.text}"}), 502
    except Exception as e:
        return jsonify({"error": f"No se pudo enviar el mensaje: {e}"}), 502

    return jsonify({"ok": True})


if __name__ == "__main__":
    # Solo para pruebas locales. En producción, Render usa gunicorn (ver Procfile / start command).
    app.run(host="0.0.0.0", port=5000, debug=True)
