FROM python:3.11-slim

# exiftool es el programa que lee los metadatos térmicos crudos de las fotos
# FLIR — no es una librería de Python, hay que instalarlo como programa del
# sistema. Por eso este servicio usa Docker en vez de la instalación simple.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "300"]
