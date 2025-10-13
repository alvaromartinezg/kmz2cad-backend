# Dockerfile
FROM python:3.11-slim

# Paquetes nativos útiles para Pillow/Matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Matplotlib headless
ENV MPLBACKEND=Agg
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

# Cloud Run expone $PORT
ENV PORT=8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
