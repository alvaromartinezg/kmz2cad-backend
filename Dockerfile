FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 libfreetype6 tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app        # <-- asegura que 'app' sea importable

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

ENV PORT=8080
EXPOSE 8080
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
