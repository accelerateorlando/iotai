import os, time, json, cv2, numpy as np, paho.mqtt.client as mqtt
from deepface import DeepFace

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
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    else:
        # assume RTSP/HTTP URL; let FFMPEG handle it
        return cv2.VideoCapture(src)

cap = open_cam(CAMERA_SRC)
if not cap or not cap.isOpened():
    raise RuntimeError(f"Cannot open camera source: {CAMERA_SRC}")
print('opened camera... deepface 3')

# --- Helpers ---
def analyze_emotion(face_img):
    """Analyze emotion using DeepFace"""
    try:
        # DeepFace expects RGB format, OpenCV uses BGR
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Analyze emotion - DeepFace returns a list of dictionaries
        result = DeepFace.analyze(
            face_rgb, 
            actions=['emotion'], 
            enforce_detection=False,  # Don't fail if face detection is poor
            detector_backend='opencv'  # Use OpenCV for consistency
        )
        
        # Extract emotion scores
        if isinstance(result, list):
            result = result[0]  # Take first result if multiple faces
        
        emotion_scores = result['emotion']
        
        # Convert to numpy array in the same order as original emotions
        # DeepFace emotions: angry, disgust, fear, happy, sad, surprise, neutral
        deepface_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        original_emotions = ["neutral","happy","surprise","sad","angry","disgust","fear","contempt"]
        
        # Map DeepFace emotions to original order
        scores = np.zeros(len(original_emotions), dtype=np.float32)
        for i, emotion in enumerate(original_emotions):
            if emotion in deepface_emotions:
                scores[i] = emotion_scores[emotion] / 100.0  # Convert percentage to 0-1
        
        # Handle contempt (not in DeepFace) - set to 0
        scores[6] = 0.0  # contempt
        
        return scores
        
    except Exception as e:
        print(f"DeepFace analysis error: {e}")
        return None

# Temporal smoothing (EMA over last n seconds)
alpha = None
last_time = 0.0
ema = np.zeros(8, dtype=np.float32)  # 8 emotions
original_emotions = ["neutral","happy","surprise","sad","angry","disgust","fear","contempt"]

while True:
    t0 = time.time()
    
    # Just do a single read
    ok, frame = cap.read()
    t1 = time.time()
    print(f'read camera {ok} - took {t1-t0:.3f}s')
    if not ok:
        time.sleep(0.5); continue
    
    # Write the captured image to a local tmp file for debugging if enabled
    SAVE_DEBUG_IMAGE = bool(os.environ.get("SAVE_DEBUG_IMAGE", "0") not in ("0", "false", "False", ""))
    if SAVE_DEBUG_IMAGE:
        debug_img_path = f"/tmp/moodcam_debug_{int(time.time())}.jpg"
        cv2.imwrite(debug_img_path, frame)
        print(f"Saved debug image to {debug_img_path}")

    # detect faces
    t2 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.2, 5)
    t3 = time.time()
    print(f'face detection took {t3-t2:.3f}s, found {len(faces)} faces')

    best_scores = None
    # pick the largest face
    if len(faces) > 0:
        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        if min(w,h) >= MIN_FACE_SIZE:
            t4 = time.time()
            crop = frame[y:y+h, x:x+w]
            
            # Analyze emotion using DeepFace
            scores = analyze_emotion(crop)
            if scores is not None:
                best_scores = scores
                t5 = time.time()
                print(f'emotion inference took {t5-t4:.3f}s')
                print(f'DeepFace scores: {scores}')
                print(f'Top emotion: {original_emotions[np.argmax(scores)]} (confidence: {np.max(scores):.4f})')

    # update smoothing
    if best_scores is not None:
        dt = t0 - last_time if last_time else 0.0
        last_time = t0
        # EMA alpha: choose so that SMOOTHING_SEC ~ time constant
        a = 1.0 - np.exp(-max(dt, 1e-3)/max(SMOOTHING_SEC, 1e-3))
        ema = (1.0 - a) * ema + a * best_scores
        top_idx = int(np.argmax(ema))
        top_emotion = original_emotions[top_idx]
        conf = float(ema[top_idx])
        print(f'EMA alpha: {a:.4f}, EMA scores: {ema}')
        print(f'Final emotion: {top_emotion} (confidence: {conf:.4f})')

        if conf >= MIN_CONFIDENCE:
            payload = {
                "camera": CAMERA_NAME,
                "emotion": top_emotion,
                "confidence": round(conf, 4),
                "scores": {k: round(float(v),4) for k,v in zip(original_emotions, ema)},
                "ts": int(time.time())
            }
            print('payload = ', str(payload))
            
            client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=False)

    # throttle loop
    t6 = time.time()
    total_time = t6 - t0
    slept = (FRAME_INTERVAL_MS/1000.0) - total_time
    print(f'total loop time: {total_time:.3f}s, sleeping {max(0, slept):.3f}s')
    if slept > 0: time.sleep(slept)
    
print('exiting... deepface')
