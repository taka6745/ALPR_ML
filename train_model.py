import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def load_annotations(annotations_path):
    annotations = []
    for filename in os.listdir(annotations_path):
        if filename.endswith('.xml'):
            filepath = os.path.join(annotations_path, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                annotations.append((root.find('filename').text,
                                    int(root.find('size').find('width').text),
                                    int(root.find('size').find('height').text),
                                    int(bbox.find('xmin').text),
                                    int(bbox.find('ymin').text),
                                    int(bbox.find('xmax').text),
                                    int(bbox.find('ymax').text)))
    return annotations

def create_training_data(annotations, images_path, target_size=(224, 224)):
    X, Y = [], []
    for filename, _, _, xmin, ymin, xmax, ymax in annotations:
        img_path = os.path.join(images_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found or cannot be opened: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        X.append(img.astype('float32') / 255.0)

        # Calculate normalized bounding box coordinates
        width, height = target_size
        norm_xmin = xmin / width
        norm_ymin = ymin / height
        norm_xmax = xmax / width
        norm_ymax = ymax / height
        Y.append([norm_xmin, norm_ymin, norm_xmax, norm_ymax])

    return np.array(X), np.array(Y)

annotations_path = './data/annotations'
images_path = './data/images'

annotations = load_annotations(annotations_path)
X, Y = create_training_data(annotations, images_path)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('license_plate_detection_best.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=10, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, monitor='val_loss')
]

model.fit(data_gen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, validation_data=(X_val, Y_val), epochs=50, callbacks=callbacks)

model.save('license_plate_detection_model.h5')
