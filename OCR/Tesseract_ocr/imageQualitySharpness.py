import cv2
import numpy as np
import pytesseract
#-------------------------------------CONTOURS-------------------------------------#

def calculate_sharpness(image):
    """
    คำนวณค่าความคมชัดของภาพโดยใช้ Laplacian variance
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

def auto_sharpen(image, target_sharpness=1000):
    """
    ปรับความคมชัดของภาพอัตโนมัติ (ปรับให้อ่อนลง)
    target_sharpness: ค่าความคมชัดที่ต้องการ (ค่าสูง = คมชัดมาก)
    """
    current_sharpness = calculate_sharpness(image)
    # print(f"ความคมชัดปัจจุบัน: {current_sharpness:.2f}")
    
    # คำนวณระดับการปรับ sharpness (ลดความแรงลง)
    if current_sharpness < 100:
        # ภาพเบลอมากๆ
        strength = 1.4
        blur_weight = -0.25
        # print("ภาพเบลอมากๆ - เพิ่มความคมชัดปานกลาง")
    elif current_sharpness < 300:
        # ภาพเบลอปานกลาง
        strength = 1.3
        blur_weight = -0.2
        # print("ภาพเบลอปานกลาง - เพิ่มความคมชัดเล็กน้อย")
    elif current_sharpness < 600:
        # ภาพเบลอเล็กน้อย
        strength = 1.2
        blur_weight = -0.15
        # print("ภาพเบลอเล็กน้อย - เพิ่มความคมชัดนิดหน่อย")
    elif current_sharpness < 1000:
        # ภาพค่อนข้างคมชัด
        strength = 1.1
        blur_weight = -0.1
        # print("ภาพค่อนข้างคมชัด - เพิ่มความคมชัดเล็กน้อยมาก")
    else:
        # ภาพคมชัดแล้ว ไม่ต้องปรับ
        strength = 1.0
        blur_weight = 0.0
        # print("ภาพคมชัดแล้ว - ไม่ต้องปรับ")
    
    if strength > 1.0:
        # สร้าง blur สำหรับ unsharp mask
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        
        # ปรับความคมชัด
        sharpened = cv2.addWeighted(image, strength, blur, blur_weight, 0)
        
        # ตรวจสอบความคมชัดหลังปรับ
        new_sharpness = calculate_sharpness(sharpened)
        # print(f"ความคมชัดหลังปรับ: {new_sharpness:.2f}")
        
        return sharpened
    else:
        return image

def remove_dot_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, blackAndWhite = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 8:  # filter small dotted regions
            img2[labels == i + 1] = 255
    res = cv2.bitwise_not(img2)
    kernel = np.ones((1, 1), np.uint8)
    res = cv2.erode(res,kernel, iterations=1)
    return res

def resize_image_to_fit_screen(img, screen_high=800, screen_width=1200):
	img_high, img_width = img.shape[:2]
	scale = min(screen_high/img_high, screen_width/img_width) 
	resized_img = cv2.resize(img, (int(img_width*scale), int(img_high*scale)))
	return resized_img, scale
#-------------------------------------CONTOURS-------------------------------------#

# Auto sharpening
def improveQuality (img) :
	sharp = auto_sharpen(img)
	# cv2.imshow('original', img)
	# cv2.imshow('auto_sharpened', sharp)
	gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('gray', gray)

	# ปรับ threshold ให้เหมาะสมกับภาพที่ผ่านการ sharpen อ่อนๆ
	ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	# cv2.imshow ('thresh', thresh)

	blank = np.zeros(img.shape, dtype=np.uint8)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(blank, contours, -1, (255,255,255), -1)
	# cv2.imshow('contours', blank)

	res_no_noise = remove_dot_noise(blank)
	# cv2.imshow('no_dot_noise', res_no_noise)
	result = sharp
     
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return result
def preprocess_date(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 7
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = remove_dot_noise(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    cv2.imshow('preprocess_date', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return th

import cv2
import numpy as np

def enhance_thai_idcard_for_ocr(img_bgr):
    """
    Input : BGR image (OpenCV)
    Output: Grayscale enhanced image (OCR-friendly)
    """

    # 1) Denoise (preserve edges)
    denoise = cv2.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2) Convert to grayscale
    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)

    # 3) Local contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )
    gray_clahe = clahe.apply(gray)

    # 4) Unsharp mask (เพิ่ม stroke ของตัวอักษร)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(
        gray_clahe, 1.5,
        blur, -0.5,
        0
    )

    # 5) Light background normalization
    bg = cv2.GaussianBlur(sharp, (31, 31), 0)
    norm = cv2.divide(sharp, bg, scale=255)
    # cv2.imshow('enhanced_thai_idcard_for_ocr', norm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return norm



if __name__ == "__main__":
	img = cv2.imread('test_data/crop3_9.jpg')
	#resize image to fit screen size
	img, _= resize_image_to_fit_screen(img)

	final_img = improveQuality(img)
	result = pytesseract.image_to_string(final_img, lang='tha', config='--psm 6 -c preserve_interword_spaces=1')
	print(result)

