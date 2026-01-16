import cv2
import json
import re
import base64
from pathlib import Path
import OCR.Tesseract_ocr.autoCorrect as ac
import OCR.Tesseract_ocr.imageQualitySharpness as iqs
import OCR.Tesseract_ocr.query_output as query_output
# import imageQualitySharpness as iqs
# import autoCorrect as ac
# import query_output
import pytesseract

current_dir = Path(__file__).parent
config_path = current_dir / 'config.json'

thai_vow = "[าิีึืุูเแโใไำ่้๊๋]"
thai_num = '[๐๑๒๓๔๕๖๗๘๙]'
eng_chr = "[abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ]"
eng_num = "[0-9]"

DATE_FIELDS = {'expireDateTH', 'expireDateEN', 'issueDateTH', 'issueDateEN','birthDateTH', 'birthDateEN'}
MODE = {'expire','issue','birth'}
BIRTHDATE_FIELDS = {'birthDateEN', 'birthDateTH'}
EXPIRE_FIELDS = {'expireDateTH', 'expireDateEN'}
ISSUE_FIELDS = {'issueDateTH', 'issueDateEN'}
ADDRESS_FIELDS = {'address1', 'address2'}
NAME_FIELDS = {'nameTH', 'nameEN'}

def replace_noise_text (text, item) :
	text = text.replace('!','').replace('\x0c','').replace('\n','').replace(',','').replace(';','').replace(':','').replace('?','').replace('\'','')
	# remove @ or any other special character that isn't - or . or /
	# special_char_pattern = r'[^ก-๙a-zA-Z0-9\s\-\./]'
	# text = re.sub(special_char_pattern, '', text)

	text = re.sub(thai_num,'',text)
	if item == 'nameEN' or item == 'nameTH' or item =='lastNameEN':
		text = re.sub(eng_num, '', text)
	if item == 'id':
		text = re.sub(eng_chr,'',text)
		text = text.replace(' ','')
	if item == 'birthDateTH' or item == 'expireDateTH' or item == 'issueDateTH' :
		thai_vow_n = re.sub('[เิี]','', thai_vow)
		text = re.sub(thai_vow_n, '', text)
	text = text.strip(' ')
	if item not in DATE_FIELDS :
		text = text.strip('-')
	# find '\n' and remove text after it
	# if '\n' in text :
	# 	text = text[:text.index('\n')]
	return text

def _load_config_data(config_path):
	try:
		with open(config_path, 'r') as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError) as exc:
		print(f"[extractData] config error: {exc}")
	except Exception as exc:
		print(f"[extractData] unexpected config error: {exc}")
	return None

def _safe_improve_quality(iqs, image, item):
	try:
		return iqs.enhance_thai_idcard_for_ocr(image)
	except Exception as exc:
		print(f"[extractData] quality error ({item}): {exc}")
		return image

def _run_ocr_text(image, lang, blacklist, item):
	blacklist = blacklist or ''
	config_text = f"--psm 6 -c preserve_interword_spaces=1 -c tessedit_char_blacklist={blacklist}"
	try:
		raw_text = pytesseract.image_to_string(image, lang=lang, config=config_text)
		return replace_noise_text(str(raw_text), item)
	except Exception as exc:
		print(f"[extractData] OCR error ({item}): {exc}")
		return ''

def _run_ocr_data(image, lang, blacklist, item):
	blacklist = blacklist or ''
	config_text = f"--psm 6 -c preserve_interword_spaces=1 -c tessedit_char_blacklist={blacklist}"
	try:
		ocr_dict = pytesseract.image_to_data(
			image,
			lang=lang,
			config=config_text,
			output_type=pytesseract.Output.DICT
		)
		texts = ocr_dict.get('text')
		if isinstance(texts, list):
			# Clean each candidate without altering alignment with confidence entries.
			ocr_dict['text'] = [replace_noise_text(str(word), item) for word in texts]
		return ocr_dict
	except Exception as exc:
		print(f"[extractData] OCR data error ({item}): {exc}")
		return {'text': [], 'conf': []}

def _compute_conf_score(conf_list):
	valid = []
	for conf in conf_list or []:
		try:
			value = float(conf)
		except (TypeError, ValueError):
			continue
		if value >= 0.0:
			valid.append(value)
	if not valid:
		return -1.0
	return sum(valid) / len(valid)

def _join_ocr_text(ocr_dict):
	if not ocr_dict:
		return ''
	words = ocr_dict.get('text', [])
	return "".join(word.strip() for word in words if word and word.strip())

def _collect_ocr_candidates(crop_img, lang, blacklist, item, iqs):
	blacklist = blacklist or ''
	enhanced_img = _safe_improve_quality(iqs, crop_img.copy(), item)
	if enhanced_img is None:
		enhanced_img = crop_img
	candidates = []
	for label, image in (('raw', crop_img), ('enhanced', enhanced_img)):
		ocr_dict = _run_ocr_data(image, lang, blacklist, item)
		# ocr_dict = _filter_low_conf(ocr_dict)
		score = _compute_conf_score(ocr_dict.get('conf', []))
		candidates.append({
			'label': label,
			'image': image,
			'ocr': ocr_dict,
			'score': score
		})
	candidates.sort(key=lambda cand: cand['score'], reverse=True)
	best = candidates[0] if candidates else {'label': 'raw', 'image': crop_img, 'ocr': {'text': [], 'conf': []}, 'score': -1.0}
	return best, candidates

def _build_date_struct(ac, text, lang, item, score):
	date = {'raw': text, 'score': score, 'alpha': '', 'numeric': ''}
	if not text:
		return date
	try:
		date['alpha'], date['numeric'], _ = ac.correct_date(text, lang=lang)
	except Exception as exc:
		print(f"[extractData] auto-correct date error ({item}): {exc}")
	return date

def _merge_twodate_results(ac, date_info, item):
	dict_th = (date_info['tha'].get('text', []), date_info['tha'].get('conf', []))
	dict_en = (date_info['eng'].get('text', []), date_info['eng'].get('conf', []))
	try:
		
		date_th, date_en = ac.correct_twodate(
			date_info['tha'].get('date', ''),
			date_info['eng'].get('date', ''),
			dict_th,
			dict_en,
			date_info['tha'].get('score', 0),
			date_info['eng'].get('score', 0)
		)
		date_info['tha']['numeric'] = date_th
		item = item.replace("DateTH", "").replace("DateEN", "")
	except Exception as exc:
		print(f"[extractData] auto-correct cross-language birthdate error: {exc}")
		date_th = (date_info['tha'].get('date', ''), date_info['tha'].get('numeric', ''))
		date_en = (date_info['eng'].get('date', ''), date_info['eng'].get('numeric', ''))
	return {
		f'{item}DateTH': {
			'raw': "".join(date_info['tha'].get('text', [])),
			'score': date_info['tha'].get('confScore', -1.0),
			'singleLangFix': date_info['tha'].get('date', ''),
			'crossLangFix': date_th[0],
			'numeric': date_th[1]
		},
		f'{item}DateEN': {
			'raw': "".join(date_info['eng'].get('text', [])),
			'score': date_info['eng'].get('confScore', -1.0),
			'singleLangFix': date_info['eng'].get('date', ''),
			'crossLangFix': date_en[0],
			'numeric': date_en[1]
		}
	}

def _handle_twodate_field(item, lang, ocr_dict, ac, info, conf_score):
	entry = {
		'text': ocr_dict.get('text', []),
		'conf': ocr_dict.get('conf', []),
		'date': '',
		'numeric': '',
		'score': 0,
		'lang': lang,
		'confScore': conf_score
	}
	try:
		entry['date'],_, entry['score'] = ac.correct_date(entry['text'], lang)
	except Exception as exc:
		print(f"[extractData] auto-correct {item} error ({item}): {exc}")
	for mode in MODE:
		if item.startswith(mode + 'Date'):
			info[mode][lang] = entry
			break
	for mode in MODE:
		if info[mode]['tha'] and info[mode]['eng'] and info[mode]['check'] == False:
			info[mode]['check'] = True
			return _merge_twodate_results(ac, info[mode],item)
	return None

def _restore_and_encode_crop(crop_img, orig_shape, resized_shape, item):
	try:
		fw = orig_shape[1] / resized_shape[1]
		fh = orig_shape[0] / resized_shape[0]
		factor = (fw + fh) / 2.0
		restored_crop = cv2.resize(crop_img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
		success, buffer = cv2.imencode('.png', restored_crop)
		if not success:
			return ''
		return base64.b64encode(buffer.tobytes()).decode('utf-8')
	except Exception as exc:
		print(f"[extractData] encode error ({item}): {exc}")
		return ''

def _match_address(ac, address_text):
	if not address_text:
		return {'raw': address_text}
	try:
		_, _, address_result = ac.match_location_from_text(address_text, threshold=0.6)
	except Exception as exc:
		print(f"[extractData] auto-correct address error: {exc}")
		address_result = {}
	address_result['raw'] = address_text
	return address_result

# def extractData(img, iqs, ac, config_path=config_path):
def extractData(img, config_path=config_path) : 
	# keep original and resized separately so we can restore crop size correctly
	# orig_img = cv2.imread(img_path)
	orig_img = img.copy()
	adjust_img = _safe_improve_quality(iqs, orig_img, 'original')
	data = {}
	try:
		resized_img, _ = iqs.resize_image_to_fit_screen(orig_img.copy())
		resized_adj_img, _ = iqs.resize_image_to_fit_screen(adjust_img.copy())
	except Exception as exc:
		print(f"[extractData] resize error: {exc}")
		return data
	img = resized_img
	adjust_img = resized_adj_img
	
	config_data = _load_config_data(config_path)
	if not config_data:
		return data

	address_parts = []
	address_scores = []
	info = {}
	for mode in MODE:
		info[mode] = {'tha': {}, 'eng': {},'check' : False}
	for item, cfg in config_data.items():
		x1,y1,x2,y2 = cfg['points']
		crop_img = img[y1:y2, x1:x2]
		# cv2.imshow(item, crop_img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		if item == 'image':
			data[item] = _restore_and_encode_crop(crop_img, orig_img.shape, img.shape, item)
			continue
		lang = cfg.get('lang', '')
		blacklist = cfg.get('black_list', '')
		best_candidate, candidates = _collect_ocr_candidates(crop_img, lang, blacklist, item, iqs)
		best_ocr = best_candidate['ocr']
		best_score = best_candidate['score']
		joined_text = _join_ocr_text(best_ocr)
		cleaned_text = replace_noise_text(joined_text, item)
		if item in NAME_FIELDS:
			raw_with_space = _run_ocr_text(best_candidate['image'], lang, blacklist, item)
			data[item] = {'raw': raw_with_space, 'score': best_score}
			continue
		if item in DATE_FIELDS:
			date_payload = _handle_twodate_field(item, lang, best_ocr, ac, info, best_score)
			if date_payload:
				data.update(date_payload)
			continue
		# if item in DATE_FIELDS:
		# 	data[item] = _build_date_struct(ac, cleaned_text, lang, item, best_score)
		# 	continue
		if item in ADDRESS_FIELDS:
			address_parts.append(cleaned_text)
			address_scores.append(best_score)
			if item == 'address2':
				full_address = " ".join(part.strip() for part in address_parts if part).strip()
				data['address'] = _match_address(ac, full_address)
				data['address']['score'] = sum(address_scores) / len(address_scores) if address_scores else -1.0
				address_parts.clear()
				address_scores.clear()
			continue
		data[item] = {'raw': cleaned_text, 'score': best_score}
	data = query_output.query_output(data)
		
	return data
# if item == 'address1'
# cv2.imshow(item, re_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def verify(test_dict, ac):
	address = ""
	data = {}
	birthDateInfoList = {'tha' : {}, 'eng' : {}}
	birthDate = {
		"text" : [],
		"conf" : [],
		"date" : "",
		"score" : 0,
		"lang" : ""
	}
	for key, value in test_dict.items():
		text = value
		item = key
		
		if item != 'image' and item != 'birthDateTH' and item != 'birthDateEN' :
			text = replace_noise_text(str(text), item)
			if item == 'expireDate' or item == 'issueDate' :
				date = {'raw' : text, 'corrected' : ''}
				date['corrected'], _ = ac.correct_date([text])  # Need to determine lang
				data[item] = date['corrected']
			elif item != 'address1' and item != 'address2' :
				data[item] = text
			
		elif item == 'birthDateEN' or item == 'birthDateTH' :
			# Handle both string and dict formats
			if isinstance(text, dict) and 'text' in text:
				birthDate['text'] = text['text']
				birthDate['conf'] = text.get('conf', [])
			else:
				# If text is a string, convert to list format
				birthDate['text'] = [str(text)]
				birthDate['conf'] = [100]  # Default confidence
			
			lang = 'tha' if item == 'birthDateTH' else 'eng'
			birthDate['date'], birthDate['score'] = ac.correct_date(birthDate['text'], lang)
			birthDate['lang'] = lang
			birthDateInfoList[lang] = birthDate.copy()
			if birthDateInfoList['tha'] != {} and birthDateInfoList['eng'] != {} :
				dictTH = (birthDateInfoList['tha']['text'], birthDateInfoList['tha']['conf'])
				dictEN = (birthDateInfoList['eng']['text'], birthDateInfoList['eng']['conf'])
				dateTH, dateEN = ac.correct_twodate(birthDateInfoList['tha']['date'], birthDateInfoList['eng']['date'], dictTH, dictEN, birthDateInfoList['tha']['score'], birthDateInfoList['eng']['score'])
				# data['birthDateTH'] = {
				# 	'raw' : "".join(birthDateInfoList['tha']['text']) if isinstance(birthDateInfoList['tha']['text'], list) else str(birthDateInfoList['tha']['text']),
				# 	'corrected' : birthDateInfoList['tha']['date'],
				# 	'corrected2' : dateTH
				# }
				# data['birthDateEN'] = {
				# 	'raw' : "".join(birthDateInfoList['eng']['text']) if isinstance(birthDateInfoList['eng']['text'], list) else str(birthDateInfoList['eng']['text']),
				# 	'corrected' : birthDateInfoList['eng']['date'],
				# 	'corrected2' : dateEN
				# }
				data['birthDateTH'] = dateTH
				data['birthDateEN'] = dateEN
		if item == 'address1' :
			address += text + " "
		if item == 'address2' :
			address += text
			_, _, address_result = ac.match_location_from_text(address, threshold=0.8)
			address_result['raw'] = address
			data['address'] = address_result
		
	return data

if __name__ == "__main__":
	data = extractData('test_data/idcard13_crop.jpg')
	for item in data:
		print(f"{item}: {data[item]}")