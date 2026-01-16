from crop_img import crop_img, read_img
import cv2
import pytesseract
from extract_data import extractData
import imageQualitySharpness as iqs
import autoCorrect as ac
import pytesseract
 
img_path = "C:\\Users\\User\\OneDrive\\Desktop\\Aomsin\\year3\\capstone\\train\\test_data\\idcard19.png"
# template_path = "./test_data/idcard_front_template.jpg"
image = read_img(img_path)
preview = crop_img(image, output_path="C:\\Users\\User\\OneDrive\\Desktop\\Aomsin\\year3\\capstone\\train\\test_data\\idcard19_crop.png")
data = extractData(img=preview)
for item in data:
	print(f"{item}: {data[item]}")
