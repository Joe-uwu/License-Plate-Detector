from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import imutils
import pytesseract
import re
import os

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        cropped_image = img[x1:x2 + 1, y1:y2 + 1]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        custom_config = r'--oem 3 --psm 8'
        text = pytesseract.image_to_string(thresh, config=custom_config)

        filtered_text = re.sub(r'[^A-Za-z0-9]', ' ', text)
        return filtered_text

    else:
        return "No license plate detected."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = f'uploads/{file.filename}'
            file.save(file_path)
            text = extract_text_from_image(file_path)
            return render_template('result.html', text=text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

