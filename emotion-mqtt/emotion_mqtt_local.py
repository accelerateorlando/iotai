import os, time, json, cv2, numpy as np, paho.mqtt.client as mqtt
import onnxruntime as ort

# --- Env ---
CAMERA_NAME = os.getenv("CAMERA_NAME","cam")
CAMERA_SRC = os.getenv("CAMERA_SRC","usb:/dev/video0")  # "usb:/dev/video0" or "rtsp://..."
MQTT_HOST = os.getenv("MQTT_HOST","localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT","1883"))
MQTT_USER = os.getenv("MQTT_USER","")
MQTT_PASS = os.getenv("MQTT_PASS","")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", f"moodcam/{CAMERA_NAME}/emotion")
FRAME_INTERVAL_MS = int(os.getenv("FRAME_INTERVAL_MS","500"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE","80"))
SMOOTHING_SEC = float(os.getenv("SMOOTHING_SEC","2.0"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE","0.6"))

# --- Model (FER+, 8 emotions) ---
# Put an FER+ ONNX file at /models/emotion-ferplus.onnx
# (labels: neutral, happy, surprise, sad, angry, disgust, fear, contempt)
MODEL_PATH = "/models/emotion-ferplus.onnx"
EMOTIONS = ["neutral","happy","surprise","sad","angry","disgust","fear","contempt"]
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# --- Face detector ---
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- MQTT ---
client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_HOST, MQTT_PORT, 60)
client.loop_start()

# --- Camera open ---
def open_cam(src: str):
    if src.startswith("usb:"):
        path = src.split("usb:")[1] or "/dev/video0"
        cap = cv2.VideoCapture(path)
        # Optional: tune capture size if your cam needs it
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    else:
        # assume RTSP/HTTP URL; let FFMPEG handle it
        return cv2.VideoCapture(src)

cap = open_cam(CAMERA_SRC)
if not cap or not cap.isOpened():
    raise RuntimeError(f"Cannot open camera source: {CAMERA_SRC}")

# --- Helpers ---
def preprocess(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64,64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img[np.newaxis, np.newaxis, :, :]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

last_time = 0.0
ema = np.zeros(len(EMOTIONS), dtype=np.float32)

while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.5)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.2, 5)
    best_scores = None

    if len(faces) > 0:
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        if min(w,h) >= MIN_FACE_SIZE:
            crop = frame[y:y+h, x:x+w]
            inp = preprocess(crop)
            out = sess.run(None, {sess.get_inputs()[0].name: inp})[0].squeeze().astype(np.float32)
            probs = softmax(out)
            best_scores = probs

    if best_scores is not None:
        dt = t0 - last_time if last_time else 0.0
        last_time = t0
        a = 1.0 - np.exp(-max(dt, 1e-3)/max(SMOOTHING_SEC, 1e-3))
        ema = (1.0 - a) * ema + a * best_scores
        idx = int(np.argmax(ema))
        mood = EMOTIONS[idx]
        conf = float(ema[idx])

        if conf >= MIN_CONFIDENCE:
            payload = {
                "camera": CAMERA_NAME,
                "emotion": mood,
                "confidence": round(conf, 4),
                "scores": {k: round(float(v),4) for k,v in zip(EMOTIONS, ema)},
                "ts": int(time.time())
            }
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=False)

    # throttle
    sleep_left = (FRAME_INTERVAL_MS/1000.0) - (time.time() - t0)
    if sleep_left > 0:
        time.sleep(sleep_left)
