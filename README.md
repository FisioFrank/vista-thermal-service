# Servicio de decodificación térmica — VIXTA

Este es un servicio pequeño y separado de tu app principal. Su único
trabajo: recibir una foto FLIR + las posiciones de las "cajas" de zona,
y devolver la temperatura real dentro de cada caja.

## Qué contiene

- `app.py` — el servicio (Flask + la librería `flyr`, que sabe leer el
  formato radiométrico de FLIR).
- `requirements.txt` — las librerías que necesita.

## Desplegarlo en Render (gratis)

Lo vamos a hacer paso a paso en el chat. Resumen de lo que vamos a hacer:

1. Subir estos archivos a un repositorio de GitHub (gratis, solo arrastrar y soltar, sin usar la terminal).
2. Conectar ese repositorio a Render.com (gratis para este uso).
3. Render instala todo automáticamente y te da una URL pública, algo como
   `https://vixta-thermal.onrender.com`.
4. Esa URL es la que VIXTA va a usar para pedir las temperaturas.

## Probarlo una vez esté desplegado

Con curl (o Postman, o cualquier cliente HTTP):

```bash
curl -X POST https://TU-URL.onrender.com/extract \
  -F "image=@foto_flir.jpg" \
  -F 'boxes=[{"id":"muslo_izq","x":0.15,"y":0.30,"width":0.20,"height":0.15}]'
```

Debería devolver algo como:

```json
{
  "width": 480,
  "height": 640,
  "results": [
    {"id": "muslo_izq", "mean": 32.4, "min": 31.1, "max": 33.8}
  ]
}
```

## Nota importante

El plan gratuito de Render "duerme" el servicio después de un rato sin
uso, y tarda unos 30-50 segundos en "despertar" la primera vez que lo
usas después de estar inactivo. Para uso real con tu equipo, quizás
valga la pena mirar un plan pago más adelante (unos pocos dólares al
mes) para que responda al instante siempre.
