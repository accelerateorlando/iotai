FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY emotion_mqtt_deepface.py /app/emotion_mqtt.py
COPY models /models

CMD ["python", "-u", "emotion_mqtt.py"]
