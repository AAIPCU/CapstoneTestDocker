# Pipeline
import yaml

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

import pandas as pd

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_pipeline(image_bytes: bytes) -> io.BytesIO:
    # Convert bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_file = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Step 1: Crop Image
    cropped_image = crop_img(image_file)

    # Step 2: Extrct Data
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
    # image_path = "C:\\Users\\user\\Desktop\\CEDT\\Capstone\\dev\\data\\spoof_dataset\\cropped\\all\\IMG_20251215_143711.jpg"
    # with open(image_path, "rb") as img_file:
    #     image_bytes = img_file.read()
    # run_pipeline(image_bytes)

    config = load_config()
    folder_path = Path(config["input_folder"])
    output_path = Path(config["output_file"])
    sample_limit = int(config["sample_limit"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame()
    file_list = list(folder_path.glob("*.*"))
    file_count = 0
    if sample_limit != -1:
        file_list = file_list[:sample_limit]
    print(f"Processing {len(file_list)} images from {folder_path} ")
    total_images = len(file_list)
    for image_path in file_list:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        key = image_path.name.split("_")[0]
        # key = image_path.name
        print(f"Processing image: {image_path.name} ({file_count+1}/{total_images})")
        result = run_pipeline(image_bytes)
        result["Test id"] = key
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        file_count += 1

    selected_columns = ["Test id","idCardNo","id_score","fullnameTh","fullnameTh_score","firstNameTh","lastNameTh","fullnameEn","firstNameEn","firstnameEn_score","lastNameEn","lastNameEn_score","Birth","birthDateEN_score","birthDateTh","birthDateTH_score","Issue","issueDateEN_score","issueDateTh","issueDateTH_score","Expire","expireDateEN_score","expiryDateTh","expireDateTH_score","addressFull","address_score","addressNo","moo","trok","soi","street","subDistrict","district","province","hologram_detected","max_hologram_confidence","spoofing_detected","spoofing_score","id_length_valid","id_location_code_valid","id_checksum_valid","requestNumber","request_length_valid","request_format_valid","request_location_code_valid","issue_date_valid"]
    results_df = results_df[selected_columns]

    results_df.to_csv(output_path, index=False)
    print(f"Pipeline results saved to {output_path}")