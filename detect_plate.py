import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

def preprocess_image(image, target_size=(224, 224)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def postprocess_predictions(predictions, original_shape):
    height, width = original_shape[:2]
    xmin, ymin, xmax, ymax = predictions[0]
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    # Ensure coordinates are within image dimensions
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    return xmin, ymin, xmax, ymax

def detect_and_crop_plate(model, image_path, save_dir):
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Failed to load image: {image_path}")
        return
    processed_img = preprocess_image(original_img)
    predictions = model.predict(processed_img)
    
    xmin, ymin, xmax, ymax = postprocess_predictions(predictions, original_img.shape)
    # Check if the bounding box is valid
    if xmax <= xmin or ymax <= ymin:
        print(f"Invalid bounding box for image: {image_path}")
        return
    
    cropped_plate = original_img[ymin:ymax, xmin:xmax]
    if cropped_plate.size == 0:
        print(f"Cropped image is empty for image: {image_path}")
        return
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"cropped_{filename}")
    cv2.imwrite(save_path, cropped_plate)
    print(f"Cropped plate saved: {save_path}")

def process_all_images(images_dir, model_path, save_dir):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}. Please check the path and ensure the model has been saved.")
    
    model = load_model(model_path, compile=False)
    
    for image_name in os.listdir(images_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(images_dir, image_name)
        detect_and_crop_plate(model, image_path, save_dir)

# Correct these paths as per your directory structure
images_dir = './data/images'
model_path = './license_plate_detection_model.h5'  # Adjust this path
save_dir = './data/cropped_images'

process_all_images(images_dir, model_path, save_dir)
