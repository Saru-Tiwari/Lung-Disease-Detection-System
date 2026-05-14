import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import os
import base64
import traceback
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ===================== CONFIG =====================
PNEUMONIA_MODEL_PATH = "D:\Personal Health assistant\Pneumonia-Detection-System\saved_res_model_dir"
TB_MODEL_PATH = "D:\Personal Health assistant\Pneumonia-Detection-System\saved_res_model_dir/resnet50_epoch_14 (1).pth"

OUTPUT_DIR = "static/"
SPRING_BOOT_ORIGIN = "http://10.10.1.149:1052"
# ==================================================

# ===================== LOAD MODELS =====================
# Pneumonia
pneumonia_model = tf.saved_model.load(PNEUMONIA_MODEL_PATH)
pneumonia_infer = pneumonia_model.signatures["serving_default"]

# Tuberculosis (PyTorch)
device = torch.device("cpu")
tb_model = models.resnet50()
tb_model.fc = torch.nn.Linear(tb_model.fc.in_features, 2)  # binary classification

checkpoint = torch.load(TB_MODEL_PATH, map_location=device, weights_only=False)
tb_model.load_state_dict(checkpoint['model_state_dict'])
tb_model.eval()
tb_model.to(device)

tb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===================== FASTAPI SETUP =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[SPRING_BOOT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== IMAGE UTILITIES =====================
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.asarray(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def overlay_bounding_box(original_image, mask, output_path,
                         max_box_ratio=0.5, min_box_area=500):
    mask_bin = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    orig_width, orig_height = original_image.size
    mask_height, mask_width = mask.shape[:2]
    scale_x = orig_width / mask_width
    scale_y = orig_height / mask_height

    localized_image = np.array(original_image)
    if len(localized_image.shape) == 2 or localized_image.shape[2] != 3:
        localized_image = cv2.cvtColor(localized_image, cv2.COLOR_GRAY2BGR)

    pneumonia_detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        image_area = mask_width * mask_height
        box_ratio = box_area / image_area

        if box_ratio < max_box_ratio and box_area > min_box_area:
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            cv2.rectangle(
                localized_image,
                (x, y),
                (x + w, y + h),
                (168, 85, 230),
                8
            )
            pneumonia_detected = True

    if pneumonia_detected:
        cv2.imwrite(output_path, localized_image)

    return pneumonia_detected, output_path if pneumonia_detected else None

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    image_array = np.array(original_image)
    if len(image_array.shape) == 2 or image_array.shape[2] != 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(image_array, 1 - alpha, heatmap_color, alpha, 0)

# ===================== GRAD-CAM FOR TB =====================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().detach().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ===================== ROUTES =====================
@app.get("/")
def root():
    return {"message": "ML Pneumonia & TB Detection Service is running"}

# --- Pneumonia API ---
@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()

        # Ensure it's an image
        try:
            image = Image.open(BytesIO(img_bytes))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        image = image.convert("RGB")  # convert all images to RGB
        original_image = image.copy()

        preprocessed_image = preprocess_image(image)
        predictions = pneumonia_infer(tf.convert_to_tensor(preprocessed_image, dtype=tf.float32))
        prediction_mask = predictions['output_0'].numpy()[0, :, :, 0]

        probabilities = tf.nn.sigmoid(predictions['output_0']).numpy()
        max_probability = np.max(probabilities)

        output_filename = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}_localized.png"
        )

        pneumonia_detected, localized_image_path = overlay_bounding_box(
            original_image, prediction_mask, output_filename
        )

        if pneumonia_detected:
            with open(localized_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            result = {
                "diagnosis": "Pneumonia",
                "probability": round(float(max_probability), 3),
                "localized_image": img_base64,
                "lung_opacity": "Present"
            }
        else:
            result = {
                "diagnosis": "No Pneumonia",
                "probability": round(float(max_probability), 3),
                "localized_image": "",
                "lung_opacity": "Absent"
            }

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Tuberculosis API ---
@app.post("/predict-tb")
async def predict_tb(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()

        # Ensure it's an image
        try:
            image = Image.open(BytesIO(img_bytes))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        image = image.convert("RGB")
        original_image = image.copy()

        input_tensor = tb_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = tb_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_names = ["Healthy", "Tuberculosis"]
        predicted_label = class_names[predicted.item()]
        probability = float(confidence.item())

        img_base64 = ""  # default: no box

        # Only generate Grad-CAM if TB is predicted
        if predicted_label == "Tuberculosis":
            cam = generate_gradcam(tb_model, input_tensor)

            # Threshold to detect hot regions
            heatmap_uint8 = np.uint8(255 * cam)
            _, thresh = cv2.threshold(heatmap_uint8, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            display_img = np.array(original_image)
            if len(display_img.shape) == 2 or display_img.shape[2] != 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            scale_x = original_image.size[0] / 224
            scale_y = original_image.size[1] / 224

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (168, 85, 230), 4)

            _, buffer = cv2.imencode(".png", display_img)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

        result = {
            "diagnosis": predicted_label,
            "probability": round(probability, 3),
            "localized_image": img_base64,
            "lung_opacity": "Present" if predicted_label == "Tuberculosis" else "Absent"
        }

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})