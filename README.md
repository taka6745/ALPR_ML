# Automatic License Plate Recognition (ALPR) System

## Overview
This project implements an ALPR system using deep learning models for license plate detection and Optical Character Recognition (OCR) to read the text from detected license plates.

The training data for license plate detection is sourced from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection). However, it has been found to be limited in quantity and quality. To improve the system, it is recommended to check the annotations on the training data and consider acquiring additional images, such as the UFPR ALPR dataset. Exploring other models like Single Shot or YOLO can also be beneficial.

To demonstrate the functionality of the system, you can use the provided `./Correct.png` image. To run the ALPR system, execute `train_model.py` to train the model and `detect_plate.py` to detect and crop license plates from new images.

Additionally, you may consider developing your own OCR model specifically for recognizing number plate letters. This can potentially enhance the accuracy of the system. It is worth noting that including Australian number plates is optional and may not be necessary for your specific use case.

To use the ALPR system, please download the required data and organize it as follows:
- Place the data in the `/data` directory.
- Store the images in the `/data/images` directory.
- Store the annotations in the `/data/annotations` directory.
- Store the cropped license plate images in the `/data/cropped_images` directory.

## Dependencies
- TensorFlow
- OpenCV
- NumPy
- PyTesseract
- Pillow

Please refer to `requirements.txt` for the detailed list of dependencies.

## Usage
1. Prepare your dataset and train the model with `train_model.py`.
2. Detect and crop license plates from new images using `detect_plate.py`.
3. Read text from cropped license plate images with `read_plate.py`.

## Portfolio
See `portfolio.json` for a summary of project metadata and performance metrics.

## License
[Your License Here]
