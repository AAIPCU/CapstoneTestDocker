# Pipeline
import sys
from pathlib import Path
pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.append(str(pipeline_dir))

import cv2
import numpy as np
import io
from OCR.Tesseract_ocr.crop_img import crop_img
from FraudDetection.hologramDetection import detect_hologram
from FraudDetection.spoofDetection import spoofCheck
from FraudDetection.infoCheck import run_info_check

from OCR.Tesseract_ocr.extract_data import extractData
from OCR.Tesseract_ocr.formatData import process_extracted_data

def run_pipeline(image_bytes: bytes) -> io.BytesIO:
    # Convert bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_file = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Step 1: Crop Image
    cropped_image = crop_img(image_file)

    # Step 2: Extract Data
    extracted_data = extractData(cropped_image)
    extracted_data = process_extracted_data(extracted_data)

    # Step 3: Hologram Detection
    extracted_data["max_hologram_confidence"], extracted_data["hologram_detected"] = detect_hologram(cropped_image)
    
    # Step 4: FFT
    extracted_data["spoofing_score"], extracted_data["spoofing_detected"] = spoofCheck(cropped_image)

    # Step 5: Information Verification
    run_info_check(extracted_data)

    for item in extracted_data:
        print(f"{item}: {extracted_data[item]}")
    
    return extracted_data

if __name__ == "__main__":
    image_path = "OCR/data/id1.png"
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    run_pipeline(image_bytes)