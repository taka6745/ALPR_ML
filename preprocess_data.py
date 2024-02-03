import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def load_annotations(annotations_path):
    annotations = []
    for filename in os.listdir(annotations_path):
        if not filename.endswith('.xml'): continue
        filepath = os.path.join(annotations_path, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            annotations.append((root.find('filename').text, xmin, ymin, xmax, ymax))
    return annotations

def create_training_data(annotations, images_path, output_size=(224, 224)):
    X, Y = [], []
    for filename, xmin, ymin, xmax, ymax in annotations:
        img_path = os.path.join(images_path, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, output_size)
        X.append(img_resized)
        Y.append([xmin/w, ymin/h, xmax/w, ymax/h])  # Normalize coords
    return np.array(X), np.array(Y)
