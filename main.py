from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import shutil
import os
import numpy as np
import json
import uuid
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ================== CONFIG ==================
API_KEY = os.getenv("API_KEY", "bezubaan_secret_0107")
MODEL_PATH = "animal_classifier_model.h5"
LABELS_PATH = "labels.json"
TEMP_DIR = "temp"

# ================== INIT ==================
app = FastAPI()

os.makedirs(TEMP_DIR, exist_ok=True)

# Load model safely
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# Load labels safely
try:
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # auto-detect format
    if isinstance(list(labels.keys())[0], str):
        labels = {v: k for k, v in labels.items()}
except Exception as e:
    print(f"❌ Labels loading failed: {e}")
    labels = {}

# ================== HELPERS ==================
def validate_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ================== ROUTES ==================

@app.get("/")
def home():
    return {"message": "Bezubaan ML API is LIVE 🚀"}

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🚨 Ensure model loaded
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{file_id}.jpg")

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate image
        if not validate_image(file_path):
            return {
                "animal": None,
                "confidence": 0.0,
                "status": "invalid_file",
                "message": "Uploaded file is not a valid image"
            }

        # Preprocess
        img_array = preprocess(file_path)

        # Prediction (faster way)
        preds = model(img_array, training=False).numpy()

        confidence = float(np.max(preds))
        class_idx = int(np.argmax(preds))

        label = labels.get(class_idx, "unknown")

        if label == "not_animal":
            return {
                "animal": None,
                "confidence": confidence,
                "status": "invalid_image"
            }

        return {
            "animal": label,
            "confidence": confidence,
            "status": "valid"
        }

    except Exception as e:
        return {
            "animal": None,
            "confidence": 0.0,
            "status": "error",
            "message": str(e)
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)