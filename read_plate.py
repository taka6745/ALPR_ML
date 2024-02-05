
import pytesseract

def read_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate_text = pytesseract.image_to_string(gray, config='--psm 7')
    
    return plate_text

