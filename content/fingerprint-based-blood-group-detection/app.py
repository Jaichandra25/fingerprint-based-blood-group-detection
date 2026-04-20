import io
import os
from typing import Tuple

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "test",
    "model_blood_group_detection_resnet.h5",
)
CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

model = None
model_load_error = None


def load_inference_model() -> None:
    global model, model_load_error
    try:
        model = load_model(MODEL_PATH)
        model_load_error = None
    except Exception as exc:
        model = None
        model_load_error = str(exc)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def crop_foreground(gray_img: Image.Image) -> Image.Image:
    arr = np.array(gray_img)
    mask = arr < 245
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return gray_img

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    pad_x = max(4, int((x_max - x_min) * 0.03))
    pad_y = max(4, int((y_max - y_min) * 0.03))

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(arr.shape[1] - 1, x_max + pad_x)
    y_max = min(arr.shape[0] - 1, y_max + pad_y)

    return gray_img.crop((x_min, y_min, x_max + 1, y_max + 1))


def preprocess_fingerprint(img: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(img)
    cropped = crop_foreground(gray)
    base = cropped.resize((256, 256), Image.Resampling.BILINEAR)

    variants = [
        base,
        ImageOps.autocontrast(base),
        ImageOps.equalize(base),
        ImageOps.autocontrast(ImageOps.invert(base)),
    ]

    batch = []
    for variant in variants:
        rgb = variant.convert("RGB")
        arr = image.img_to_array(rgb).astype(np.float32)
        batch.append(arr)

    batch_array = np.array(batch, dtype=np.float32)
    return preprocess_input(batch_array)


def predict_blood_group(img: Image.Image) -> Tuple[str, float]:
    if model is None:
        raise RuntimeError("Model not loaded")

    batch = preprocess_fingerprint(img)
    predictions = model.predict(batch, verbose=0)
    mean_prediction = np.mean(predictions, axis=0)
    predicted_idx = int(np.argmax(mean_prediction))
    confidence = float(np.max(mean_prediction))

    if predicted_idx < 0 or predicted_idx >= len(CLASSES):
        raise RuntimeError("Invalid prediction index from model")
    if not np.isfinite(confidence):
        raise RuntimeError("Invalid confidence from model")

    return CLASSES[predicted_idx], confidence


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    if model is None:
        return (
            jsonify(
                {
                    "error": "model_load_failure",
                    "message": model_load_error or "Failed to load model",
                }
            ),
            500,
        )

    if "image" not in request.files:
        return (
            jsonify(
                {
                    "error": "missing_image",
                    "message": "No image file provided. Use multipart field name 'image'.",
                }
            ),
            400,
        )

    uploaded = request.files["image"]
    if uploaded.filename is None or uploaded.filename.strip() == "":
        return (
            jsonify({"error": "missing_image", "message": "No selected image file."}),
            400,
        )

    if not allowed_file(uploaded.filename):
        return (
            jsonify(
                {
                    "error": "invalid_image",
                    "message": "Unsupported file type. Use png, jpg, jpeg, or bmp.",
                }
            ),
            400,
        )

    try:
        raw = uploaded.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        return (
            jsonify({"error": "invalid_image", "message": "Uploaded file is not a valid image."}),
            400,
        )
    except Exception as exc:
        return (
            jsonify({"error": "invalid_image", "message": f"Failed to read image: {exc}"}),
            400,
        )

    try:
        blood_group, confidence = predict_blood_group(img)
    except RuntimeError as exc:
        msg = str(exc)
        if "Model not loaded" in msg:
            return jsonify({"error": "model_load_failure", "message": msg}), 500
        if "Invalid" in msg:
            return jsonify({"error": "prediction_failure", "message": msg}), 500
        return jsonify({"error": "preprocessing_failure", "message": msg}), 400
    except Exception as exc:
        return jsonify({"error": "prediction_failure", "message": f"Prediction failed: {exc}"}), 500

    return jsonify({"blood_group": blood_group, "confidence": confidence})


load_inference_model()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))