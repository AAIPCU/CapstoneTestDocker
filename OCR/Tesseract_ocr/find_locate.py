import cv2

#find pixel location
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path

current_dir = Path(__file__).parent
config_path = current_dir / 'config.json'

first = (0,0)
second = (0,0)
status = 0
def find_locate_with_mouse(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		global first, second, status
		print(f"Pixel location: x={x}, y={y}")
		print(f"Real Pixel location: x={int(x/scale)}, y={int(y/scale)}")
		#mark dot on image
		cv2.circle(img, (x,y), 1, (0,255,0), 2)
		if status == 0 :
			first = (x,y)
			status = 1
		else :
			second = (x, y)
			cv2.rectangle(img,first, second,(0,255,0),1 )
			status = 0
		cv2.imshow('image', img)

def adjust_scale(screen_high, screen_width, img_high, img_width):
	scale_high = screen_high / img_high
	scale_width = screen_width / img_width
	scale = min(scale_high, scale_width)
	return scale


img = cv2.imread("C:\\Users\\User\\OneDrive\\Desktop\\Aomsin\\year3\\1-2568\\capstone\\train\\test_data\\idcard15_crop.jpg")
screen_high = 800
screen_width = 1200

img_high, img_width = img.shape[:2]
#scale = adjust_scale(screen_high, screen_width, img_high, img_width)
#want image size to be 800*1200
scale = min(800/img_high, 1200/img_width) 
img = cv2.resize(img, (int(img_width*scale), int(img_high*scale)))


data = {}
if os.path.exists(config_path):
	with open(config_path, 'r') as f:
		data = json.load(f)
	print('image size', img.shape)
	for item in data:
		x1,y1,x2,y2 = data[item]['points']
		cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0),1)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
while(True):
	mode = int(input("1: nameTH\n2: nameEN\n3: lastNameEN\n4: idnumber\n5: birthDateTH\n6: birthDateEN\n7: address1\n8: address2\n9: requestNumber\n10: image\n11: expireDateTH\n12: expireDateEN\n13: issueDateTH\n14: issueDateEN\n15: exit"))
	
	if mode == 15:
		break
	data_model = {
		'points' : [],
		'lang' : "",
		'white_list' : "",
		'black_list' : "",
		'config':"",
	}
	cv2.imshow('image', img)
	cv2.setMouseCallback('image', find_locate_with_mouse)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	data_model["points"] = [first[0], first[1], second[0], second[1]]
	print(data)
	if mode == 1:
		data_model["lang"] = "tha"
		data_model['black_list'] = "0123456789/-"
		data['nameTH'] = data_model.copy()
	elif mode == 2:
		data_model["lang"] = "eng"
		data_model['black_list'] = "0123456789-/"
		data['nameEN'] = data_model.copy()
	elif mode == 3:
		data_model["lang"] = "eng"
		data_model['black_list'] = "0123456789-./"
		data['lastNameEN'] = data_model.copy()
	elif mode == 4:
		data_model["lang"] = "eng"
		data_model['black_list'] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-/,"
		data['id'] = data_model.copy()
	elif mode == 5:
		data_model["lang"] = "tha"
		data_model['black_list'] = ","
		data['birthDateTH'] = data_model.copy()
	elif mode == 6:
		data_model["lang"] = "eng"
		data_model['black_list'] = ","
		data['birthDateEN'] = data_model.copy()
	elif mode == 7:
		data_model["lang"] = "tha"
		data_model['black_list'] = ","
		data['address1'] = data_model.copy()
	elif mode == 8:
		data_model["lang"] = "tha"
		data_model['black_list'] = ","
		data['address2'] = data_model.copy()
	elif mode == 9:
		data_model["lang"] = "eng"
		data_model['black_list'] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ./"
		data['requestNumber'] = data_model.copy()
	elif mode == 10:
		data_model["lang"] = ""
		data_model['black_list'] = ""
		data['image'] = data_model.copy()
	elif mode == 11:
		data_model["lang"] = "tha"
		data_model['black_list'] = ","
		data['expireDate'] = data_model.copy()
	elif mode == 12:
		data_model["lang"] = "eng"
		data_model['black_list'] = ","
		data['expireDateEN'] = data_model.copy()
	elif mode == 13:
		data_model["lang"] = "tha"
		data_model['black_list'] = ","
		data['issueDateTH'] = data_model.copy()
	elif mode == 14:
		data_model["lang"] = "eng"
		data_model['black_list'] = ","
		data['issueDateEN'] = data_model.copy()
	first = (0,0)
	second = (0,0)

with open("services/backend/Pipeline/OCR/Tesseract_ocr/config.json", 'w') as f:
	json.dump(data, f, indent=4)
