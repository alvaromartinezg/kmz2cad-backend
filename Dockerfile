FROM python:3.11-slim

# Paquetes nativos que necesita Matplotlib + fuentes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libfreetype6 fontconfig fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Matplotlib sin GUI y con cache en /tmp (Cloud Run solo permite escribir /tmp)
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
# (si usas PLANTILLA en /app/app, la copias ahí; si la usas en /app, muévela)
# COPY app/assets/PLANTILLA.dxf /app/PLANTILLA.dxf

ENV PORT=8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
